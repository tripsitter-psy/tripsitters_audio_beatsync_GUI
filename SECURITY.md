# Security Policy and HTTP URL Allowlist

This repository contains some vendor-supplied and legacy files with non-HTTPS (http://) URLs. These are present for legal, provenance, or compatibility reasons and must not be altered in the original vendor text. To satisfy security scanners, the following HTTP URLs are explicitly allowlisted:

## Allowlisted HTTP URLs (by file)

### tools/onnxruntime-official/onnxruntime-win-x64-1.18.1/ThirdPartyNotices.txt
- http://3rdpartysource.microsoft.com
- http://www.apache.org/licenses/
- http://www.apache.org/licenses/LICENSE-2.0
- http://mozilla.org/MPL/2.0/
- http://llvm.org
- http://docs.tvm.ai/contribute/community.html
- http://homes.cs.washington.edu/~moreau/
- http://homes.cs.washington.edu/~haichen/
- http://unlicense.org/
- http://www.opensource.org/licenses/mit-license.php
- http://en.wikipedia.org/wiki/MIT_License
- http://www.google.com/codesearch/p?hl=en#dR3YEbitojA/COPYING&q=GetSystemTimeAsFileTime%20license:bsd

### src/backend/tracing.cpp
- http://localhost:4317

### scripts/sign-windows-binaries.ps1
- http://timestamp.digicert.com

### tools/tracing/start-tracing-collector.sh, tools/tracing/start-tracing-collector.ps1, docs/TRACING.md
- http://localhost:16686
- http://localhost:4318

### Info.plist.in and Info.plist (macOS)
- http://www.apple.com/DTDs/PropertyList-1.0.dtd

## Policy
- Do not modify vendor-supplied legal notices or license files.
- All above URLs are allowlisted for security scanning purposes only.
- Any new non-HTTPS URLs must be reviewed and added to this list with justification.

For questions, contact the repository maintainers.
