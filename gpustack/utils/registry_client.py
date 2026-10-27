"""Minimal client for listing the tags of a repository on an OCI/Docker registry.

Speaks the Distribution API (``GET /v2/<name>/tags/list``), which quay.io and a
self-hosted ``registry`` both serve, so the same call lists upstream releases
online and an internal mirror offline.
"""

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional
from urllib.parse import urljoin

import httpx

logger = logging.getLogger(__name__)

_MAX_PAGES = 50
_PAGE_SIZE = 1000
_NEXT_LINK = re.compile(r'<([^>]+)>\s*;\s*rel="?next"?')
_CHALLENGE_PARAM = re.compile(r'(\w+)="([^"]*)"')


@dataclass(frozen=True)
class RepositoryRef:
    scheme: str
    host: str
    name: str

    @property
    def base_url(self) -> str:
        return f"{self.scheme}://{self.host}"


def parse_repository(repository: str) -> RepositoryRef:
    """Split ``[http(s)://]host[:port]/path/name`` into its parts.

    A bare reference means https. An internal registry served over plain HTTP
    has to say so with an explicit ``http://`` prefix.
    """
    scheme = "https"
    ref = repository.strip()
    for prefix in ("https://", "http://"):
        if ref.startswith(prefix):
            scheme = prefix[:-3]
            ref = ref[len(prefix) :]
            break
    host, sep, name = ref.rstrip("/").partition("/")
    if not host or not sep or not name:
        raise ValueError(
            f"Invalid repository '{repository}': expected host[:port]/namespace/name."
        )
    return RepositoryRef(scheme=scheme, host=host, name=name)


def image_repository(repository: str) -> str:
    """The repository as a container runtime reference, without any scheme."""
    ref = parse_repository(repository)
    return f"{ref.host}/{ref.name}"


def _bearer_token(
    client: httpx.Client,
    challenge: str,
    name: str,
    credentials: Optional[httpx.BasicAuth],
) -> Optional[str]:
    params: Dict[str, str] = dict(_CHALLENGE_PARAM.findall(challenge))
    realm = params.get("realm")
    if not realm:
        return None
    query = {"scope": params.get("scope") or f"repository:{name}:pull"}
    if params.get("service"):
        query["service"] = params["service"]
    resp = client.get(realm, params=query, auth=credentials)
    resp.raise_for_status()
    body = resp.json()
    return body.get("token") or body.get("access_token")


def list_tags(
    repository: str,
    *,
    username: Optional[str] = None,
    password: Optional[str] = None,
    timeout: float = 10.0,
    client: Optional[httpx.Client] = None,
) -> List[str]:
    """All tags of ``repository``, following pagination.

    Raises:
        httpx.HTTPError: the registry is unreachable or refuses the request.
    """
    ref = parse_repository(repository)
    credentials = httpx.BasicAuth(username, password) if username and password else None
    owns_client = client is None
    http = client or httpx.Client(timeout=timeout, follow_redirects=True)
    try:
        headers: Dict[str, str] = {}
        tags: List[str] = []
        url: Optional[str] = f"{ref.base_url}/v2/{ref.name}/tags/list?n={_PAGE_SIZE}"
        for _ in range(_MAX_PAGES):
            if url is None:
                break
            resp = http.get(url, headers=headers, auth=credentials)
            if resp.status_code == 401 and "authorization" not in headers:
                challenge = resp.headers.get("www-authenticate", "")
                if challenge.lower().startswith("bearer"):
                    token = _bearer_token(http, challenge, ref.name, credentials)
                    if token:
                        headers["authorization"] = f"Bearer {token}"
                        continue
            resp.raise_for_status()
            tags.extend(resp.json().get("tags") or [])
            link = _NEXT_LINK.search(resp.headers.get("link", ""))
            url = urljoin(url, link.group(1)) if link else None
        else:
            logger.warning(
                "Stopped listing tags of %s after %d pages", repository, _MAX_PAGES
            )
        return tags
    finally:
        if owns_client:
            http.close()
