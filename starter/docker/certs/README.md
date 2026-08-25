# Corporate TLS-inspection certificates (local only)

This machine sits behind a Zscaler TLS-inspecting proxy, which re-signs traffic to
`download.pytorch.org` and `huggingface.co`. Containers do not trust that root, so pip
and huggingface_hub fail with `CERTIFICATE_VERIFY_FAILED: unable to get local issuer
certificate` during the image build.

Any `*.crt` placed here is installed into the image trust store at build time. The
directory is intentionally empty in git — GitHub Actions runners are not behind the
proxy, so the build there is a no-op.

Regenerate on macOS:

    security find-certificate -a -c "Zscaler" -p /Library/Keychains/System.keychain \
      > docker/certs/zscaler-root.crt

`*.crt` is git-ignored: the proxy CA is specific to this network and does not belong in
a public repository.
