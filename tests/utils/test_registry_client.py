import httpx
import pytest

from gpustack.utils.registry_client import image_repository, list_tags, parse_repository


@pytest.mark.parametrize(
    "repository, scheme, host, name",
    [
        ("quay.io/ascend/vllm-ascend", "https", "quay.io", "ascend/vllm-ascend"),
        (
            "https://quay.io/ascend/vllm-ascend",
            "https",
            "quay.io",
            "ascend/vllm-ascend",
        ),
        ("http://reg.lan:5000/a/b/", "http", "reg.lan:5000", "a/b"),
    ],
)
def test_parse_repository(repository, scheme, host, name):
    ref = parse_repository(repository)
    assert (ref.scheme, ref.host, ref.name) == (scheme, host, name)


@pytest.mark.parametrize("bad", ["", "quay.io", "quay.io/", "/ascend/x"])
def test_parse_repository_rejects_a_reference_without_host_and_name(bad):
    with pytest.raises(ValueError):
        parse_repository(bad)


def test_image_repository_drops_the_scheme():
    assert image_repository("http://reg.lan:5000/a/b") == "reg.lan:5000/a/b"


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_list_tags_follows_the_next_link():
    pages = {
        "/v2/ascend/vllm-ascend/tags/list": (
            ["a", "b"],
            '</v2/ascend/vllm-ascend/tags/list?n=2&last=b>; rel="next"',
        ),
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("last") == "b":
            return httpx.Response(200, json={"tags": ["c"]})
        tags, link = pages[request.url.path]
        return httpx.Response(200, json={"tags": tags}, headers={"link": link})

    with _client(handler) as client:
        assert list_tags("quay.io/ascend/vllm-ascend", client=client) == ["a", "b", "c"]


def test_list_tags_answers_a_bearer_challenge_anonymously():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.host, request.url.path))
        if request.url.host == "auth.example":
            assert request.url.params["scope"] == "repository:ns/img:pull"
            assert request.url.params["service"] == "reg.example"
            return httpx.Response(200, json={"token": "t0k"})
        if request.headers.get("authorization") != "Bearer t0k":
            return httpx.Response(
                401,
                headers={
                    "www-authenticate": 'Bearer realm="https://auth.example/token",'
                    'service="reg.example",scope="repository:ns/img:pull"'
                },
            )
        return httpx.Response(200, json={"tags": ["v1"]})

    with _client(handler) as client:
        assert list_tags("reg.example/ns/img", client=client) == ["v1"]
    assert ("auth.example", "/token") in seen


def test_list_tags_sends_basic_credentials_when_configured():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"].startswith("Basic ")
        return httpx.Response(200, json={"tags": None})

    with _client(handler) as client:
        assert list_tags("reg.lan/a/b", username="u", password="p", client=client) == []


def test_list_tags_raises_when_the_registry_refuses():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403)

    with _client(handler) as client, pytest.raises(httpx.HTTPStatusError):
        list_tags("reg.lan/a/b", client=client)


def test_list_tags_uses_the_scheme_given_in_the_reference():
    urls = []

    def handler(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        return httpx.Response(200, json={"tags": []})

    with _client(handler) as client:
        list_tags("http://reg.lan:5000/a/b", client=client)
    assert urls[0].startswith("http://reg.lan:5000/v2/a/b/tags/list")
