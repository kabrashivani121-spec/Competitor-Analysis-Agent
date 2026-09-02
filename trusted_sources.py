"""Trusted-source policy, filtering, and report ingestion."""

from __future__ import annotations

import ipaddress
import socket
from io import BytesIO
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader


TRUSTED_PUBLISHERS = {
    # Regulators, filings, exchanges, and multilateral institutions.
    "sec.gov": "regulatory filing",
    "ftc.gov": "regulator",
    "justice.gov": "regulator",
    "europa.eu": "government / regulator",
    "ec.europa.eu": "government / regulator",
    "gov.uk": "government / regulator",
    "companieshouse.gov.uk": "company registry",
    "oecd.org": "multilateral research",
    "worldbank.org": "multilateral research",
    "imf.org": "multilateral research",
    "iea.org": "multilateral industry research",
    "acea.auto": "recognized industry association",
    "vda.de": "recognized industry association",
    "oica.net": "recognized industry association",
    "euroncap.com": "recognized independent testing body",
    "nasdaq.com": "recognized exchange",
    "nyse.com": "recognized exchange",
    "londonstockexchange.com": "recognized exchange",
    "moex.com": "recognized exchange",
    # Global investment banks and recognized equity/market research publishers.
    "goldmansachs.com": "investment-bank research",
    "morganstanley.com": "investment-bank research",
    "jpmorgan.com": "investment-bank research",
    "ubs.com": "investment-bank research",
    "bofa.com": "investment-bank research",
    "business.bofa.com": "investment-bank research",
    "citi.com": "investment-bank research",
    "citigroup.com": "investment-bank research",
    "barclays.com": "investment-bank research",
    "db.com": "investment-bank research",
    "jefferies.com": "investment-bank research",
    "evercore.com": "investment-bank research",
    "lazard.com": "investment-bank research",
    # Ratings, consulting, and established industry research.
    "spglobal.com": "ratings / industry research",
    "moodys.com": "ratings / industry research",
    "fitchratings.com": "ratings / industry research",
    "mckinsey.com": "recognized consulting research",
    "bcg.com": "recognized consulting research",
    "bain.com": "recognized consulting research",
    "deloitte.com": "recognized professional research",
    "pwc.com": "recognized professional research",
    "ey.com": "recognized professional research",
    "kpmg.com": "recognized professional research",
    "gartner.com": "recognized industry research",
    "forrester.com": "recognized industry research",
    "idc.com": "recognized industry research",
    "morningstar.com": "recognized investment research",
    "cfraresearch.com": "recognized investment research",
    # Established business reporting and recognized product-review sources.
    "reuters.com": "recognized business press",
    "bloomberg.com": "recognized business press",
    "ft.com": "recognized business press",
    "wsj.com": "recognized business press",
    "economist.com": "recognized business press",
    "g2.com": "recognized customer-review platform",
    "capterra.com": "recognized customer-review platform",
    "jdpower.com": "recognized industry and customer research",
    "consumerreports.org": "recognized independent product research",
    "edmunds.com": "recognized automotive research and reviews",
}


def normalize_domain(value: str) -> str:
    candidate = value.strip().lower()
    if "://" not in candidate:
        candidate = "https://" + candidate
    hostname = urlparse(candidate).hostname or ""
    return hostname.removeprefix("www.").rstrip(".")


def domain_matches(domain: str, allowed: str) -> bool:
    return domain == allowed or domain.endswith("." + allowed)


def classify_trusted_url(
    url: str,
    user_domains: list[str] | tuple[str, ...] = (),
    official_domains: list[str] | tuple[str, ...] = (),
) -> str | None:
    domain = normalize_domain(url)
    if not domain:
        return None
    for allowed in official_domains:
        if domain_matches(domain, normalize_domain(allowed)):
            return "official company / competitor website"
    for allowed in user_domains:
        if domain_matches(domain, normalize_domain(allowed)):
            return "user-approved report or official source"
    for allowed, category in TRUSTED_PUBLISHERS.items():
        if domain_matches(domain, allowed):
            return category
    if domain.endswith(".gov") or domain.endswith(".gov.au") or domain.endswith(".gc.ca"):
        return "government / regulator"
    if domain.endswith(".edu") or domain.endswith(".ac.uk"):
        return "recognized academic research"
    return None


def trusted_organic_results(
    results: dict,
    user_domains: list[str] | tuple[str, ...] = (),
) -> tuple[list[dict], list[str]]:
    """Keep only recognized publishers and an official site from the knowledge graph."""
    official_domains: list[str] = []
    website = (results.get("knowledge_graph") or {}).get("website", "")
    if website:
        official_domains.append(normalize_domain(website))

    accepted = []
    rejected = []
    for item in results.get("organic_results", []):
        url = item.get("link", "")
        category = classify_trusted_url(url, user_domains, official_domains)
        if not category:
            if url:
                rejected.append(normalize_domain(url))
            continue
        accepted.append({**item, "trust_classification": category})
    return accepted, sorted(set(filter(None, rejected)))


def _validate_public_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Only public HTTP(S) report URLs are supported")
    if parsed.port not in {None, 80, 443}:
        raise ValueError("Only standard HTTP(S) ports are supported")
    if parsed.hostname.lower() == "localhost":
        raise ValueError("Local addresses cannot be fetched")
    try:
        addresses = socket.getaddrinfo(parsed.hostname, parsed.port or 443)
    except socket.gaierror as exc:
        raise ValueError(f"Unable to resolve source hostname: {parsed.hostname}") from exc
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if not ip.is_global:
            raise ValueError("Private or local source addresses cannot be fetched")
    return url


def _download_public_source(url: str, max_bytes: int = 8_000_000) -> tuple[bytes, str, str]:
    current = _validate_public_url(url)
    headers = {"User-Agent": "BenchmarkingStudio/1.0 research-source-loader"}
    for _ in range(4):
        response = requests.get(
            current,
            headers=headers,
            timeout=25,
            stream=True,
            allow_redirects=False,
        )
        if response.is_redirect:
            target = response.headers.get("location")
            if not target:
                raise ValueError("Source returned an invalid redirect")
            current = _validate_public_url(urljoin(current, target))
            continue
        response.raise_for_status()
        content = bytearray()
        for chunk in response.iter_content(64 * 1024):
            content.extend(chunk)
            if len(content) > max_bytes:
                raise ValueError("Source exceeds the 8 MB ingestion limit")
        return bytes(content), response.headers.get("content-type", ""), current
    raise ValueError("Source redirected too many times")


def extract_report_text(name: str, content: bytes, content_type: str = "") -> str:
    is_pdf = name.lower().endswith(".pdf") or "application/pdf" in content_type.lower()
    if is_pdf:
        reader = PdfReader(BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages[:80])[:50_000]
    decoded = content.decode("utf-8", errors="replace")
    if "html" in content_type.lower() or "<html" in decoded[:500].lower():
        soup = BeautifulSoup(decoded, "html.parser")
        for element in soup(["script", "style", "nav", "footer"]):
            element.decompose()
        decoded = soup.get_text("\n", strip=True)
    return decoded[:50_000]


def build_trusted_source_context(
    urls: list[str],
    uploaded_files: list[tuple[str, bytes]],
) -> tuple[str, list[str], list[str]]:
    """Fetch explicit trusted URLs and extract uploaded reports for prompt context."""
    user_domains = []
    sections = []
    errors = []
    for url in urls:
        try:
            content, content_type, final_url = _download_public_source(url)
            text = extract_report_text(final_url, content, content_type)
            if not text.strip():
                raise ValueError("No readable text found")
            sections.append(f"### User-approved source: {final_url}\n{text[:15_000]}")
            user_domains.append(normalize_domain(final_url))
        except Exception as exc:
            errors.append(f"{url}: {exc}")
    for name, content in uploaded_files:
        try:
            if len(content) > 8_000_000:
                raise ValueError("Uploaded report exceeds the 8 MB ingestion limit")
            text = extract_report_text(name, content)
            if not text.strip():
                raise ValueError("No readable text found")
            sections.append(f"### User-provided report: {name}\n{text[:20_000]}")
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    return "\n\n".join(sections)[:45_000], sorted(set(filter(None, user_domains))), errors
