from __future__ import annotations

import importlib
import os
import sys
from typing import TypedDict


class SPDXLicense(TypedDict):
    id: str
    deprecated: bool


class SPDXException(TypedDict):
    id: str
    deprecated: bool


def _load_bundled_data() -> tuple[dict[str, SPDXLicense], dict[str, SPDXException]] | None:
    module_names = (
        "pip._vendor.packaging.licenses._spdx",
        "setuptools._vendor.packaging.licenses._spdx",
        "wheel.vendored.packaging.licenses._spdx",
    )
    for name in module_names:
        try:
            module = importlib.import_module(name)
            licenses = getattr(module, "LICENSES")
            exceptions = getattr(module, "EXCEPTIONS")
            if isinstance(licenses, dict) and isinstance(exceptions, dict):
                return dict(licenses), dict(exceptions)
        except Exception:
            pass

    current = os.path.realpath(globals().get("__file__", ""))
    for entry in sys.path:
        try:
            candidate = os.path.join(entry, "packaging", "licenses", "_spdx.py")
            if not os.path.isfile(candidate):
                continue
            if os.path.realpath(candidate) == current:
                continue
            namespace: dict[str, object] = {
                "__name__": "_packaging_spdx_data",
                "__file__": candidate,
                "__package__": "packaging.licenses",
            }
            with open(candidate, "rb") as source:
                exec(compile(source.read(), candidate, "exec"), namespace)
            licenses = namespace.get("LICENSES")
            exceptions = namespace.get("EXCEPTIONS")
            if isinstance(licenses, dict) and isinstance(exceptions, dict):
                return dict(licenses), dict(exceptions)
        except Exception:
            pass
    return None


_data = _load_bundled_data()

if _data is not None:
    LICENSES, EXCEPTIONS = _data
else:
    _license_ids = """
0BSD
3D-Slicer-1.0
AAL
Abstyles
AdaCore-doc
Adobe-2006
Adobe-Display-PostScript
Adobe-Glyph
Adobe-Utopia
ADSL
AFL-1.1
AFL-1.2
AFL-2.0
AFL-2.1
AFL-3.0
Afmparse
AGPL-1.0
AGPL-1.0-only
AGPL-1.0-or-later
AGPL-3.0
AGPL-3.0-only
AGPL-3.0-or-later
Aladdin
AMD-newlib
AMDPLPA
AML
AML-glslang
AMPAS
ANTLR-PD
ANTLR-PD-fallback
any-OSI
any-OSI-perl-modules
Apache-1.0
Apache-1.1
Apache-2.0
APAFML
APL-1.0
App-s2p
APSL-1.0
APSL-1.1
APSL-1.2
APSL-2.0
Arphic-1999
Artistic-1.0
Artistic-1.0-cl8
Artistic-1.0-Perl
Artistic-2.0
Artistic-dist
Aspell-RU
ASWF-Digital-Assets-1.0
ASWF-Digital-Assets-1.1
Baekmuk
Bahyph
Barr
bcrypt-Solar-Designer
Beerware
Bitstream-Charter
Bitstream-Vera
BitTorrent-1.0
BitTorrent-1.1
blessing
BlueOak-1.0.0
Boehm-GC
Boehm-GC-without-fee
Borceux
Brian-Gladman-2-Clause
Brian-Gladman-3-Clause
BSD-1-Clause
BSD-2-Clause
BSD-2-Clause-Darwin
BSD-2-Clause-first-lines
BSD-2-Clause-FreeBSD
BSD-2-Clause-NetBSD
BSD-2-Clause-Patent
BSD-2-Clause-pkgconf-disclaimer
BSD-2-Clause-Views
BSD-3-Clause
BSD-3-Clause-acpica
BSD-3-Clause-Attribution
BSD-3-Clause-Clear
BSD-3-Clause-flex
BSD-3-Clause-HP
BSD-3-Clause-LBNL
BSD-3-Clause-Modification
BSD-3-Clause-No-Military-License
BSD-3-Clause-No-Nuclear-License
BSD-3-Clause-No-Nuclear-License-2014
BSD-3-Clause-No-Nuclear-Warranty
BSD-3-Clause-Open-MPI
BSD-3-Clause-Sun
BSD-4-Clause
BSD-4-Clause-Shortened
BSD-4-Clause-UC
BSD-4.3RENO
BSD-4.3TAHOE
BSD-Advertising-Acknowledgement
BSD-Attribution-HPND-disclaimer
BSD-Inferno-Nettverk
BSD-Protection
BSD-Source-beginning-file
BSD-Source-Code
BSD-Systemics
BSD-Systemics-W3Works
BSL-1.0
BUSL-1.1
bzip2-1.0.5
bzip2-1.0.6
C-UDA-1.0
CAL-1.0
CAL-1.0-Combined-Work-Exception
Caldera
Caldera-no-preamble
Catharon
CATOSL-1.1
CC-BY-1.0
CC-BY-2.0
CC-BY-2.5
CC-BY-2.5-AU
CC-BY-3.0
CC-BY-3.0-AT
CC-BY-3.0-AU
CC-BY-3.0-DE
CC-BY-3.0-IGO
CC-BY-3.0-NL
CC-BY-3.0-US
CC-BY-4.0
CC-BY-NC-1.0
CC-BY-NC-2.0
CC-BY-NC-2.5
CC-BY-NC-3.0
CC-BY-NC-3.0-DE
CC-BY-NC-4.0
CC-BY-NC-ND-1.0
CC-BY-NC-ND-2.0
CC-BY-NC-ND-2.5
CC-BY-NC-ND-3.0
CC-BY-NC-ND-3.0-DE
CC-BY-NC-ND-4.0
CC-BY-NC-SA-1.0
CC-BY-NC-SA-2.0
CC-BY-NC-SA-2.5
CC-BY-NC-SA-3.0
CC-BY-NC-SA-3.0-DE
CC-BY-NC-SA-4.0
CC-BY-ND-1.0
CC-BY-ND-2.0
CC-BY-ND-2.5
CC-BY-ND-3.0
CC-BY-ND-3.0-DE
CC-BY-ND-4.0
CC-BY-SA-1.0
CC-BY-SA-2.0
CC-BY-SA-2.5
CC-BY-SA-3.0
CC-BY-SA-3.0-AT
CC-BY-SA-4.0
CC0-1.0
CDDL-1.0
CDDL-1.1
CDL-1.0
CDLA-Permissive-1.0
CDLA-Permissive-2.0
CDLA-Sharing-1.0
CECILL-1.0
CECILL-1.1
CECILL-2.0
CECILL-2.1
CECILL-B
CECILL-C
CERN-OHL-1.1
CERN-OHL-1.2
CERN-OHL-P-2.0
CERN-OHL-S-2.0
CERN-OHL-W-2.0
ClArtistic
CLISP-exception-2.0
CNRI-Jython
CNRI-Python
CNRI-Python-GPL-Compatible
CPAL-1.0
CPL-1.0
CPOL-1.02
Crossword
CrystalStacker
CUA-OPL-1.0
Cube
curl
D-FSL-1.0
diffmark
DOC
Dotseqn
DSDP
dvipdfm
ECL-1.0
ECL-2.0
EFL-1.0
EFL-2.0
eGenix
Entessa
EPICS
EPL-1.0
EPL-2.0
ErlPL-1.1
etalab-2.0
EUDatagrid
EUPL-1.0
EUPL-1.1
EUPL-1.2
Eurosym
Fair
Ferguson-Twofish
Frameworx-1.0
FreeBSD-DOC
FreeImage
FSFAP
FSFUL
FSFULLR
FSFMIT
FTL
GFDL-1.1-invariants-only
GFDL-1.1-invariants-or-later
GFDL-1.1-no-invariants-only
GFDL-1.1-no-invariants-or-later
GFDL-1.1-only
GFDL-1.1-or-later
GFDL-1.2-invariants-only
GFDL-1.2-invariants-or-later
GFDL-1.2-no-invariants-only
GFDL-1.2-no-invariants-or-later
GFDL-1.2-only
GFDL-1.2-or-later
GFDL-1.3-invariants-only
GFDL-1.3-invariants-or-later
GFDL-1.3-no-invariants-only
GFDL-1.3-no-invariants-or-later
GFDL-1.3-only
GFDL-1.3-or-later
Giftware
GL2PS
Glide
Glulxe
GLWTPL
GNAT-exception
GNUplot
GPL-1.0
GPL-1.0-only
GPL-1.0-or-later
GPL-2.0
GPL-2.0-only
GPL-2.0-or-later
GPL-3.0
GPL-3.0-only
GPL-3.0-or-later
gSOAP-1.3b
HaskellReport
Hippocratic-2.1
HPND
HPND-doc
HPND-export-US
HPND-export-US-acknowledgement
HPND-Fenneberg-Livingston
HPND-INRIA-IMAG
HPND-Intel
HPND-Kevlin-Henney
HPND-Markus-Kuhn
HPND-MIT-disclaimer
HPND-Netrek
HPND-Pbmplus
HPND-sell-variant
HPND-sell-variant-MIT-disclaimer
HPND-UC
HTMLTIDY
IBM-pibs
ICU
IEC-Code-Components-EULA
IJG
ImageMagick
iMatix
Imlib2
Info-ZIP
Intel
Intel-ACPI
Interbase-1.0
IPA
IPL-1.0
ISC
JasPer-2.0
JPL-image
JPNIC
JSON
Kastrup
Kazlib
Knuth-CTAN
LAL-1.2
LAL-1.3
Latex2e
Leptonica
LGPL-2.0
LGPL-2.0-only
LGPL-2.0-or-later
LGPL-2.1
LGPL-2.1-only
LGPL-2.1-or-later
LGPL-3.0
LGPL-3.0-only
LGPL-3.0-or-later
LGPLLR
Libpng
libpng-2.0
libselinux-1.0
libtiff
Linux-man-pages-1-para
Linux-man-pages-copyleft
Linux-man-pages-copyleft-2-para
Linux-man-pages-copyleft-var
Linux-OpenIB
LOOP
LPPL-1.0
LPPL-1.1
LPPL-1.2
LPPL-1.3a
LPPL-1.3c
MakeIndex
MirOS
MIT
MIT-0
MIT-advertising
MIT-CMU
MIT-enna
MIT-feh
MIT-Modern-Variant
MIT-open-group
MITNFA
Motosoto
MPL-1.0
MPL-1.1
MPL-2.0
MPL-2.0-no-copyleft-exception
MS-PL
MS-RL
MTLL
MulanPSL-1.0
MulanPSL-2.0
Multics
Mup
NASA-1.3
Naumen
NBPL-1.0
NCGL-UK-2.0
NCSA
Net-SNMP
NetCDF
Newsletr
NGPL
NIST-PD
NIST-PD-fallback
NLOD-1.0
NLPL
Nokia
NOSL
Noweb
NPL-1.0
NPL-1.1
NPOSL-3.0
NRL
NTP
NTP-0
OCCT-PL
OCLC-2.0
ODbL-1.0
OFL-1.0
OFL-1.1
OGC-1.0
OGDL-Taiwan-1.0
OGL-Canada-2.0
OGL-UK-1.0
OGL-UK-2.0
OGL-UK-3.0
OGTSL
OLDAP-1.1
OLDAP-1.2
OLDAP-1.3
OLDAP-1.4
OLDAP-2.0
OLDAP-2.0.1
OLDAP-2.1
OLDAP-2.2
OLDAP-2.2.1
OLDAP-2.2.2
OLDAP-2.3
OLDAP-2.4
OLDAP-2.5
OLDAP-2.6
OLDAP-2.7
OLDAP-2.8
OML
OpenSSL
OPL-1.0
OSET-PL-2.1
OSL-1.0
OSL-1.1
OSL-2.0
OSL-2.1
OSL-3.0
Parity-6.0.0
Parity-7.0.0
PDDL-1.0
PHP-3.0
PHP-3.01
Plexus
PolyForm-Noncommercial-1.0.0
PolyForm-Small-Business-1.0.0
PostgreSQL
PSF-2.0
psfrag
psutils
Python-2.0
Python-2.0.1
Qhull
QPL-1.0
Rdisc
RHeCos-1.1
RPL-1.1
RPL-1.5
RPSL-1.0
RSA-MD
RSCPL
Ruby
SAX-PD
Saxpath
SCEA
SchemeReport
Sendmail
Sendmail-8.23
SGI-B-1.0
SGI-B-1.1
SGI-B-2.0
SHL-0.5
SHL-0.51
SISSL
SISSL-1.2
Sleepycat
SMLNJ
SMPPL
SNIA
Spencer-86
Spencer-94
Spencer-99
SPL-1.0
SSH-OpenSSH
SSH-short
SSPL-1.0
SugarCRM-1.1.3
SWL
TAPR-OHL-1.0
TCL
TCP-wrappers
TMate
TORQUE-1.1
TOSL
TPDL-1.0
TU-Berlin-1.0
TU-Berlin-2.0
UCOS
Unicode-3.0
Unicode-DFS-2015
Unicode-DFS-2016
Unicode-TOU
Unlicense
UPL-1.0
Vim
VOSTROM
VSL-1.0
W3C
W3C-19980720
W3C-20150513
w3m
Watcom-1.0
Wsuipa
WTFPL
X11
Xerox
XFree86-1.1
xinetd
Xnet
xpp
XSkat
YPL-1.0
YPL-1.1
Zed
Zend-2.0
Zimbra-1.3
Zimbra-1.4
Zlib
zlib-acknowledgement
ZPL-1.1
ZPL-2.0
ZPL-2.1
""".split()

    _deprecated_licenses = {
        "AGPL-1.0",
        "AGPL-3.0",
        "BSD-2-Clause-FreeBSD",
        "BSD-2-Clause-NetBSD",
        "bzip2-1.0.5",
        "GPL-1.0",
        "GPL-2.0",
        "LGPL-2.0",
        "LGPL-2.1",
    }

    LICENSES: dict[str, SPDXLicense] = {
        identifier.lower(): {
            "id": identifier,
            "deprecated": identifier in _deprecated_licenses,
        }
        for identifier in _license_ids
    }

    _exception_ids = """
389-exception
Asterisk-exception
Autoconf-exception-2.0
Autoconf-exception-3.0
Bison-exception-2.2
Bootloader-exception
Classpath-exception-2.0
CLISP-exception-2.0
Cryptsetup-OpenSSL-exception
DigiRule-FOSS-exception
eCos-exception-2.0
Elastic-2.0
Fawkes-Runtime-exception
FLTK-exception
Font-exception-2.0
freertos-exception-2.0
GCC-exception-2.0
GCC-exception-3.1
gnu-javamail-exception
GPL-3.0-interface-exception
GPL-3.0-linking-exception
GPL-3.0-linking-source-exception
GPL-CC-1.0
GStreamer-exception-2005
GStreamer-exception-2008
i2p-gpl-java-exception
LGPL-3.0-linking-exception
Libtool-exception
Linux-syscall-note
LZMA-exception
mif-exception
Nokia-Qt-exception-1.1
OCaml-LGPL-linking-exception
OpenJDK-assembly-exception-1.0
OpenVPN-openssl-exception
PS-or-PDF-font-exception-20170817
Qt-GPL-exception-1.0
Qt-LGPL-exception-1.1
Qwt-exception-1.0
SHL-2.0
SHL-2.1
Solderpad-0.5
SWI-exception
u-boot-exception-2.0
Universal-FOSS-exception-1.0
UPL-1.0
WxWindows-exception-3.1
x11vnc-openssl-exception
""".split()

    EXCEPTIONS: dict[str, SPDXException] = {
        identifier.lower(): {"id": identifier, "deprecated": False}
        for identifier in _exception_ids
    }

VERSION = "3.27.0"

del _data
try:
    del _license_ids, _deprecated_licenses, _exception_ids
except NameError:
    pass