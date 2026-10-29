# Licences (recorded 2026-10-21T00:30:42Z, BEFORE installing or downloading any package or browser)

Sources: metadata saved in data_untrusted/metadata/ (PyPI JSON, GitHub API JSON, docs pages), fetched with curl on 2026-10-21.

| Component | Version seen | Licence | Source of the licence fact |
|---|---|---|---|
| MiniWoB++ (pip: miniwob) | 1.1.0 (PyPI upload 2026-08-13T18:41Z; GitHub release v1.1 2026-08-13, "Maintenance release"; repo not archived, last push 2026-08-13) | MIT (Copyright 2016 OpenAI, 2018 Stanford, 2022 Farama Foundation) | GitHub /license API (spdx MIT) + decoded LICENSE text; PyPI classifier "License :: OSI Approved :: MIT License" |
| Playwright for Python | latest 1.64.0 (2026-10-13) | Apache-2.0 | PyPI license_expression; GitHub microsoft/playwright-python spdx |
| Selenium (Python) | latest 4.51.0 (2026-10-09) | Apache-2.0 | PyPI license; GitHub SeleniumHQ/selenium spdx |
| Chromium / chromium-headless-shell (browser that Playwright downloads) | set by the Playwright version | Chromium source: BSD-3-Clause ("Copyright 2015 The Chromium Authors") plus bundled third-party licences | chromium.googlesource.com/chromium/src/+/main/LICENSE |
| Chrome for Testing chrome / chromedriver / chrome-headless-shell (Selenium route) | matched to the browser version | The CfT README (tooling repo is Apache-2.0) and the CfT blog page give no separate binary licence. Free public download from storage.googleapis.com, no account, no key. Built from Chromium (BSD-3-Clause); check the licence files inside the zips after download | GoogleChromeLabs/chrome-for-testing README; developer.chrome.com/blog/chrome-for-testing |

Cost: all free; no account, no API key, no paid service.
Supply-chain rule: resolve Python packages with uv --exclude-newer 2026-10-07T00:00:00Z (14-day cooldown) and record wheel SHA256 hashes (requirements.lock.txt).
Python dependency licences are listed after install (from installed package metadata).

## Found inside the downloads (recorded 2026-10-21T00:33:44Z)
- Chrome for Testing 153.0.8010.12 (Playwright "chromium" and "chromium-headless-shell"): ABOUT file = "Google Chrome. Copyright 2026 Google LLC. All rights reserved. Chrome is made possible by the Chromium open source project and other open source software. See the Terms of Service at chrome://terms." So the binaries are under the Google Chrome Terms of Service (free, no account), with open-source parts.
- chrome-headless-shell ships LICENSE.headless_shell: Chromium BSD-3-Clause + all third-party licences (42,329 lines).
- Full CfT chrome bundled WidevineCdm with a proprietary LICENSE ("not open source software", no use without a separate agreement). That folder was deleted; nothing here needs it.

## Python packages installed in .venv (from installed metadata, 2026-10-21T00:43:20Z)

| Package | Version | Licence (metadata) |
|---|---|---|
| attrs | 26.1.0 | MIT |
| certifi | 2026.7.22 | Mozilla Public License 2.0 (MPL 2.0) |
| cloudpickle | 3.1.2 | BSD License |
| Farama-Notifications | 0.0.6 | MIT License |
| greenlet | 3.5.6 | MIT AND PSF-2.0 |
| gymnasium | 1.4.0 | MIT License |
| h11 | 0.16.0 | MIT License |
| idna | 3.20 | BSD-3-Clause |
| miniwob | 1.1.0 | MIT License |
| numpy | 2.5.3 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| outcome | 1.3.0.post0 | MIT License; Apache Software License |
| pillow | 12.3.0 | MIT-CMU |
| playwright | 1.63.0 | Apache-2.0 |
| pyee | 13.0.1 | MIT License |
| PySocks | 1.7.1 | BSD |
| selenium | 4.50.0 | Apache-2.0 |
| sniffio | 1.3.1 | MIT License; Apache Software License |
| sortedcontainers | 2.4.0 | Apache Software License |
| trio | 0.34.0 | MIT OR Apache-2.0 |
| trio-websocket | 0.12.2 | MIT License |
| typing_extensions | 4.16.0 | PSF-2.0 |
| urllib3 | 2.8.0 | MIT |
| websocket-client | 1.9.2 | Apache-2.0 |
| wsproto | 1.3.2 | MIT |

Pinned with SHA256 hashes in requirements.lock.txt (uv pip compile --generate-hashes --exclude-newer 2026-10-07T00:00:00Z).

## conda-forge userspace libraries for the browser (recorded 2026-10-21T01:20:23Z, BEFORE the install)

Source of each licence fact: api.anaconda.org package metadata saved on 2026-10-21 in data_untrusted/metadata/condaforge/<name>.json ("license" field). Free downloads from conda.anaconda.org/conda-forge, no account, no key, no cost. All are open-source licences (LGPL / MPL / MIT-type / Apache / AFL-or-GPL). Transitive dependencies are listed after the install, from each installed package's conda-meta record.

| Package (requested) | Latest seen | Licence | Needed for |
|---|---|---|---|
| libglib | 2.90.1 | LGPL-2.1-or-later | libglib-2.0, libgobject-2.0, libgio-2.0 (headless shell, chromedriver) |
| nss | 3.118 | MPL-2.0 | libnss3, libnssutil3, libsmime3 |
| nspr | 4.40 | MPL-2.0 | libnspr4 |
| atk-1.0 | 2.38.0 | LGPL-2.0-or-later | libatk-1.0 |
| at-spi2-atk | 2.38.0 | LGPL-2.1 | libatk-bridge-2.0 |
| at-spi2-core | 2.58.3 | LGPL-2.1-or-later | libatspi |
| dbus | 1.16.2 | AFL-2.1 OR GPL-2.0-or-later | libdbus-1 |
| xorg-libx11 | 1.8.13 | MIT | libX11 |
| xorg-libxcomposite | 0.4.7 | MIT | libXcomposite |
| xorg-libxdamage | 1.1.6 | MIT | libXdamage |
| xorg-libxext | 1.3.7 | MIT | libXext |
| xorg-libxfixes | 6.0.2 | MIT | libXfixes |
| xorg-libxrandr | 1.5.5 | MIT | libXrandr |
| libxcb | 1.17.0 | MIT | libxcb |
| libxkbcommon | 1.13.2 | MIT AND MIT-open-group AND HPND AND HPND-sell-variant AND ISC | libxkbcommon |
| libgbm | 1.0.9 | MIT | libgbm |
| libdrm | 2.4.129 | MIT | libdrm (gbm dependency) |
| alsa-lib | 1.2.16.1 | LGPL-2.1-or-later | libasound |
| fontconfig | 2.18.3 | MIT | font configuration (no /etc/fonts here) |
| fonts-conda-ecosystem | 1 | BSD-3-Clause (meta package; the font files' own licences are listed after install) | fonts |
| cairo | 1.18.6 | LGPL-2.1-only or MPL-1.1 | full CfT chrome only |
| pango | 1.58.2 | LGPL-2.1-or-later | full CfT chrome only |
| libcups | 2.3.3 | Apache-2.0 | full CfT chrome only |

Use: only the browser process and its driver get LD_LIBRARY_PATH / FONTCONFIG_FILE (wrapper scripts or Playwright's launch env); nothing system-wide.

### Full solved set: 70 conda-forge packages (recorded 2026-10-21T01:29:08Z from the repodata 'license' field, BEFORE the install)

Solved offline against conda-forge repodata filtered to builds on or before 2026-10-07T00:00:00Z (the same 14-day cooldown as the PyPI lock). Lock file with URL + sha256 per package: browser-libs.explicit.lock.txt. Every licence below is a free open-source licence (or a free font licence); no proprietary or paid component. Runtime libraries under GPL-3.0 WITH GCC-exception-3.1 (libgcc, libstdcxx, libgomp) are the usual GCC runtime exception.

| Package | Version | Build | Built (UTC date) | Licence (repodata) |
|---|---|---|---|---|
| _openmp_mutex | 4.5 | 20_gnu | 2026-02-12 | BSD-3-Clause |
| alsa-lib | 1.2.16.1 | h7cc23a3_1 | 2026-08-26 | LGPL-2.1-or-later |
| at-spi2-atk | 2.38.0 | h0630a04_3 | 2021-04-22 | LGPL-2.1-or-later |
| at-spi2-core | 2.40.3 | h0630a04_0 | 2021-07-09 | LGPL-2.1-or-later |
| atk-1.0 | 2.38.0 | h04ea711_2 | 2024-04-23 | LGPL-2.0-or-later |
| bzip2 | 1.0.8 | hda65f42_10 | 2026-08-05 | bzip2-1.0.6 |
| ca-certificates | 2026.7.22 | hbd8a1cb_0 | 2026-07-22 | ISC |
| cairo | 1.18.6 | h3c89d7e_0 | 2026-09-22 | LGPL-2.1-only or MPL-1.1 |
| dbus | 1.16.2 | he8c428d_2 | 2026-08-31 | AFL-2.1 OR GPL-2.0-or-later |
| expat | 2.8.5 | hd2095e1_0 | 2026-09-25 | MIT |
| font-ttf-dejavu-sans-mono | 2.37 | hab24e00_0 | 2019-08-27 | BSD-3-Clause |
| font-ttf-inconsolata | 3.000 | h77eed37_0 | 2021-05-08 | OFL-1.1 |
| font-ttf-source-code-pro | 2.038 | h77eed37_0 | 2021-05-08 | OFL-1.1 |
| font-ttf-ubuntu | 0.83 | h77eed37_3 | 2024-09-28 | LicenseRef-Ubuntu-Font-Licence-Version-1.0 |
| fontconfig | 2.18.3 | h4db4eae_1 | 2026-08-14 | MIT |
| fonts-conda-ecosystem | 1 | 0 | 2019-08-28 | BSD-3-Clause |
| fonts-conda-forge | 1 | hc364b38_1 | 2025-11-05 | BSD-3-Clause |
| fribidi | 1.0.17 | h7cc23a3_0 | 2026-09-21 | LGPL-2.1-or-later |
| graphite2 | 1.3.15 | h54a6638_1 | 2026-08-07 | LGPL-2.0-or-later |
| icu | 78.3 | py310h44b86e0_2 | 2026-08-12 | MIT |
| keyutils | 1.6.3 | h7cc23a3_1 | 2026-08-14 | LGPL-2.1-or-later |
| krb5 | 1.22.2 | hbc21106_2 | 2026-08-15 | MIT |
| libcups | 2.3.3 | h7a8fb5f_6 | 2026-02-12 | Apache-2.0 |
| libdrm | 2.4.129 | h7cc23a3_0 | 2026-08-14 | MIT |
| libedit | 3.1.20260512 | pl5321h373387f_0 | 2026-10-06 | BSD-2-Clause |
| libexpat | 2.8.5 | hd2095e1_0 | 2026-09-25 | MIT |
| libffi | 3.7.0 | h81df57d_1 | 2026-08-26 | MIT |
| libfreetype | 2.14.3 | ha770c72_2 | 2026-08-13 | GPL-2.0-only OR FTL |
| libfreetype6 | 2.14.3 | h5e6c136_2 | 2026-08-13 | GPL-2.0-only OR FTL |
| libgbm | 1.0.9 | hee9eb32_0 | 2026-09-25 | MIT |
| libgcc | 16.2.0 | ha9f2e26_7 | 2026-09-25 | GPL-3.0-only WITH GCC-exception-3.1 |
| libgcc-ng | 16.2.0 | h69a702a_7 | 2026-09-25 | GPL-3.0-only WITH GCC-exception-3.1 |
| libglib | 2.90.0 | h569388d_0 | 2026-09-15 | LGPL-2.1-or-later |
| libgomp | 16.2.0 | he0feb66_7 | 2026-09-25 | GPL-3.0-only WITH GCC-exception-3.1 |
| libharfbuzz | 14.6.0 | h23af247_0 | 2026-10-06 | MIT |
| libiconv | 1.18 | h0cb94f2_3 | 2026-08-18 | LGPL-2.1-only |
| liblzma | 5.8.3 | hb03c661_1 | 2026-08-10 | 0BSD |
| libpciaccess | 0.19 | hb03c661_1 | 2026-08-05 | MIT |
| libpng | 1.6.59 | h922cc85_0 | 2026-09-30 | zlib-acknowledgement |
| libsqlite | 3.53.4 | h0737f62_102 | 2026-10-05 | blessing |
| libstdcxx | 16.2.0 | h934c35e_7 | 2026-09-25 | GPL-3.0-only WITH GCC-exception-3.1 |
| libstdcxx-ng | 16.2.0 | hdf11a46_7 | 2026-09-25 | GPL-3.0-only WITH GCC-exception-3.1 |
| libuuid | 2.42.4 | hcfc3c73_0 | 2026-09-22 | BSD-3-Clause |
| libxcb | 1.17.0 | hb83e432_2 | 2026-08-28 | MIT |
| libxkbcommon | 1.13.2 | h51789e4_1 | 2026-08-19 | MIT AND MIT-open-group AND HPND AND HPND-sell-variant AND ISC |
| libxml2 | 2.15.4 | h7df9aa5_0 | 2026-09-05 | MIT |
| libxml2-16 | 2.15.4 | hf3af7cc_0 | 2026-09-05 | MIT |
| libzlib | 1.3.2 | h25fd6f3_3 | 2026-07-29 | Zlib |
| ncurses | 6.6 | hdb14827_1 | 2026-08-10 | X11 AND BSD-3-Clause |
| nspr | 4.40 | h29cc59b_0 | 2026-08-08 | MPL-2.0 |
| nss | 3.118 | h445c969_0 | 2025-11-18 | MPL-2.0 |
| openssl | 3.6.5 | h781a0a9_0 | 2026-10-04 | Apache-2.0 |
| pango | 1.58.2 | h7bb47b9_1 | 2026-09-04 | LGPL-2.1-or-later |
| pcre2 | 10.47 | h8b3dc9c_1 | 2026-08-21 | BSD-3-Clause |
| pixman | 0.46.4 | h54a6638_3 | 2026-08-07 | MIT |
| pthread-stubs | 0.4 | h7cc23a3_1004 | 2026-09-02 | MIT |
| xkeyboard-config | 2.48 | h280c20c_0 | 2026-06-21 | MIT |
| xorg-libice | 1.1.2 | h280c20c_0 | 2026-08-11 | MIT |
| xorg-libsm | 1.2.6 | h0d788c3_1 | 2026-08-12 | MIT |
| xorg-libx11 | 1.8.13 | he1eb515_1 | 2026-08-18 | MIT |
| xorg-libxau | 1.0.12 | h7cc23a3_2 | 2026-09-09 | MIT |
| xorg-libxcomposite | 0.4.7 | hb03c661_0 | 2026-02-02 | MIT |
| xorg-libxdamage | 1.1.6 | hb9d3cd8_0 | 2024-10-02 | MIT |
| xorg-libxdmcp | 1.1.5 | h7cc23a3_2 | 2026-09-17 | MIT |
| xorg-libxext | 1.3.7 | h7cc23a3_1 | 2026-08-19 | MIT |
| xorg-libxfixes | 6.0.2 | h7cc23a3_1 | 2026-08-20 | MIT |
| xorg-libxi | 1.8.3 | h7cc23a3_1 | 2026-08-20 | MIT |
| xorg-libxrandr | 1.5.5 | h7cc23a3_1 | 2026-08-20 | MIT |
| xorg-libxrender | 0.9.12 | hb03c661_1 | 2026-08-19 | MIT |
| xorg-libxtst | 1.2.5 | h7cc23a3_4 | 2026-08-22 | MIT |
