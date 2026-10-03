"""One irreversible request reservation per endpoint slot, then the frozen guard."""
from shinka_crl.adaptive_endpoint import provider_main


if __name__ == "__main__":
    raise SystemExit(provider_main())
