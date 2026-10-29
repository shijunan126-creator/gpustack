"""Turn off TLS verification inside aisbench's HTTP client on request.

aisbench creates its ``aiohttp`` connector without an SSL argument, so the
platform's choice not to verify the target (GPUStack's worker-wide insecure
TLS setting, passed as ``verify: false``) has no way in. The inference task
runs in a separate Python process, so the switch is an environment variable
the runner sets for the aisbench process and this module, loaded at
interpreter start by ``aisbench_runner_tls.pth``, applies there.
"""

import os


def apply() -> None:
    if os.environ.get("AISBENCH_RUNNER_TLS_INSECURE") != "1":
        return
    import aiohttp

    original = aiohttp.TCPConnector.__init__

    def init(self, *args, **kwargs):
        kwargs.setdefault("ssl", False)
        original(self, *args, **kwargs)

    aiohttp.TCPConnector.__init__ = init
