"""One irreversible request reservation per continuation slot, then the frozen guard."""
from shinka_crl.adaptive_continuation import provider_main


if __name__ == "__main__":
    raise SystemExit(provider_main())
