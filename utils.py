"""
Utility functions for Competitor Analysis System
Includes PDF export and data formatting helpers
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional
from io import BytesIO
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak,
    Table, TableStyle, KeepTogether
)
from reportlab.lib import colors
import re

logger = logging.getLogger(__name__)


DETAILED_ANALYSIS_DIMENSIONS = (
    "Financial Performance",
    "Business Model",
    "Product/Services",
    "Pricing Structure",
    "Brand & Marketing",
    "Sales & Distribution",
    "Market Reach/Share",
    "Customer Perception",
    "Operational Capabilities",
    "Talent & Culture",
    "Strategic Moves",
    "SWOT Analysis",
)

QUANTIFIED_DETAILED_ANALYSIS_DIMENSIONS = (
    "Financial Performance",
    "Product/Services",
    "Pricing Structure",
    "Sales & Distribution",
    "Market Reach/Share",
)

DETAILED_ANALYSIS_FRAMEWORK = {
    "Assess": ("Industry Overview", "Competitor Landscape"),
    "Benchmark (Competitor Deep-Dive)": DETAILED_ANALYSIS_DIMENSIONS,
    "Strategize": ("Competitive Strategy", "Actionable Recommendations"),
}


class PDFReportGenerator:
    """Generate professional PDF reports from competitor analysis results"""
    
    def __init__(self, company_name: str, industry: str):
        """
        Initialize PDF generator
        
        Args:
            company_name: Name of the company being analyzed
            industry: Industry sector
        """
        self.company_name = company_name
        self.industry = industry
        self.buffer = BytesIO()
        self.styles = getSampleStyleSheet()
        self._create_custom_styles()
        
    def _create_custom_styles(self):
        """Create custom paragraph styles for the report"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1a1a1a'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        # Heading 2 style
        self.styles.add(ParagraphStyle(
            name='CustomHeading2',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        ))
        
        # Heading 3 style
        self.styles.add(ParagraphStyle(
            name='CustomHeading3',
            parent=self.styles['Heading3'],
            fontSize=14,
            textColor=colors.HexColor('#34495e'),
            spaceAfter=10,
            spaceBefore=10,
            fontName='Helvetica-Bold'
        ))
        
        # Body text style
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['BodyText'],
            fontSize=11,
            leading=16,
            textColor=colors.HexColor('#2c3e50'),
            alignment=TA_JUSTIFY,
            spaceAfter=10
        ))
        
        # Bullet point style
        self.styles.add(ParagraphStyle(
            name='CustomBullet',
            parent=self.styles['BodyText'],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor('#2c3e50'),
            leftIndent=20,
            spaceAfter=6
        ))
    
    def generate_pdf(self, report_content: str) -> BytesIO:
        """
        Generate PDF from report content
        
        Args:
            report_content: Full report text content
            
        Returns:
            BytesIO: PDF file buffer
        """
        try:
            logger.info(f"Generating PDF report for {self.company_name}")
            
            # Create PDF document
            doc = SimpleDocTemplate(
                self.buffer,
                pagesize=letter,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )
            
            # Build content elements
            elements = []
            
            # Add title page
            elements.extend(self._create_title_page())
            
            # Parse and add report content
            elements.extend(self._parse_report_content(report_content))
            
            # Build PDF
            doc.build(elements)
            
            # Reset buffer position
            self.buffer.seek(0)
            
            logger.info("PDF generated successfully")
            return self.buffer
            
        except Exception as e:
            logger.error(f"Error generating PDF: {str(e)}")
            raise
    
    def _create_title_page(self) -> List:
        """Create title page elements"""
        elements = []
        
        # Title
        title = Paragraph(
            "COMPETITOR ANALYSIS REPORT",
            self.styles['CustomTitle']
        )
        elements.append(title)
        elements.append(Spacer(1, 0.3 * inch))
        
        # Company name
        company = Paragraph(
            f"<b>{self.company_name}</b>",
            self.styles['CustomHeading2']
        )
        elements.append(company)
        elements.append(Spacer(1, 0.2 * inch))
        
        # Industry
        industry = Paragraph(
            f"Industry: {self.industry}",
            self.styles['CustomBody']
        )
        elements.append(industry)
        elements.append(Spacer(1, 0.1 * inch))
        
        # Date
        date = Paragraph(
            f"Report Date: {datetime.now().strftime('%B %d, %Y')}",
            self.styles['CustomBody']
        )
        elements.append(date)
        
        # Add page break
        elements.append(PageBreak())
        
        return elements
    
    def _parse_report_content(self, content: str) -> List:
        """Parse markdown-style report content into PDF elements"""
        elements = []
        lines = content.split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines
            if not line:
                i += 1
                continue
            
            # Handle headers
            if line.startswith('# '):
                # H1
                text = line[2:].strip()
                elements.append(Paragraph(text, self.styles['CustomTitle']))
                elements.append(Spacer(1, 0.2 * inch))
                
            elif line.startswith('## '):
                # H2
                text = line[3:].strip()
                elements.append(Spacer(1, 0.15 * inch))
                elements.append(Paragraph(text, self.styles['CustomHeading2']))
                elements.append(Spacer(1, 0.1 * inch))
                
            elif line.startswith('### '):
                # H3
                text = line[4:].strip()
                elements.append(Paragraph(text, self.styles['CustomHeading3']))
                elements.append(Spacer(1, 0.05 * inch))
                
            elif line.startswith('- ') or line.startswith('* '):
                # Bullet point
                text = line[2:].strip()
                bullet = Paragraph(f"• {text}", self.styles['CustomBullet'])
                elements.append(bullet)
                
            elif re.match(r'^\d+\.\s+', line):
                # Numbered list
                num, text = line.split('.', 1)
                text = text.strip()
                bullet = Paragraph(f"{num}. {text}", self.styles['CustomBullet'])
                elements.append(bullet)
                
            elif line.startswith('---'):
                # Horizontal rule
                elements.append(Spacer(1, 0.1 * inch))
                
            else:
                # Regular paragraph
                if line:
                    # Clean up markdown formatting
                    text = self._clean_markdown(line)
                    para = Paragraph(text, self.styles['CustomBody'])
                    elements.append(para)
            
            i += 1
        
        return elements
    
    def _clean_markdown(self, text: str) -> str:
        """Clean markdown formatting for PDF rendering"""
        # Bold
        text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
        # Italic
        text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
        # Code (inline)
        text = re.sub(r'`(.+?)`', r'<font name="Courier">\1</font>', text)
        
        return text


def format_report_for_display(report_content: str) -> Dict[str, str]:
    """
    Parse report content into sections for display in Streamlit tabs
    
    Args:
        report_content: Full report text
        
    Returns:
        Dict mapping section names to content
    """
    sections = {
        "overview": "",
        "detailed_analysis": "",
        "competitor_matrix": "",
        "recommendations": ""
    }
    
    try:
        # Split content into sections based on headers
        current_section = None
        current_content = []

        def flush_section():
            nonlocal current_content
            if not current_section or not current_content:
                return

            content = '\n'.join(current_content).strip()
            if not content:
                current_content = []
                return

            if sections[current_section]:
                sections[current_section] += f"\n\n{content}"
            else:
                sections[current_section] = content
            current_content = []
        
        lines = report_content.split('\n')
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("## ") and not stripped.startswith("### "):
                heading = stripped[3:].strip().lower()
                if heading in {
                    "executive summary",
                    "key findings",
                    "competitive landscape overview",
                }:
                    target_section = "overview"
                elif heading == "detailed competitor analysis":
                    target_section = "detailed_analysis"
                elif "comparison matrix" in heading:
                    target_section = "competitor_matrix"
                elif heading in {
                    "market opportunities",
                    "competitive threats",
                    "strategic recommendations",
                    "conclusion",
                    "trusted source register",
                }:
                    target_section = "recommendations"
                else:
                    target_section = current_section

                if target_section != current_section:
                    flush_section()
                    current_section = target_section
                if current_section:
                    current_content.append(line)
            elif current_section:
                current_content.append(line)
        
        # Add final section
        flush_section()
        
        # If sections are empty, put everything in overview
        if not any(sections.values()):
            sections["overview"] = report_content
            
    except Exception as e:
        logger.error(f"Error formatting report: {str(e)}")
        sections["overview"] = report_content
    
    return sections


def extract_key_metrics(report_content: str) -> Dict:
    """
    Extract key metrics from the report for summary display
    
    Args:
        report_content: Full report text
        
    Returns:
        Dict containing extracted metrics
    """
    metrics = {
        "num_competitors": 0,
        "key_findings": [],
        "threat_level": "Medium",
        "opportunities": 0
    }
    
    try:
        current_h2 = ""
        for raw_line in report_content.splitlines():
            line = raw_line.strip()
            line_lower = line.lower()

            if line.startswith('## '):
                current_h2 = line[3:].strip().lower()
                continue

            if (
                'detailed competitor analysis' in current_h2
                and line.startswith('### ')
            ):
                metrics["num_competitors"] += 1

            list_match = re.match(r'^(?:\d+\.|[-*])\s+(.+)', line)
            if list_match and 'key findings' in current_h2:
                metrics["key_findings"].append(list_match.group(1).strip())
            if list_match and 'market opportunities' in current_h2:
                metrics["opportunities"] += 1

            plain_line = re.sub(r'[*_`]', '', line_lower)
            threat_match = re.search(
                r'competitive threat level\s*:?\s*(high|medium|low)',
                plain_line,
            )
            if threat_match:
                levels = {'low': 1, 'medium': 2, 'high': 3}
                current = metrics["threat_level"].lower()
                candidate = threat_match.group(1)
                if levels[candidate] > levels[current]:
                    metrics["threat_level"] = candidate.title()
                    
    except Exception as e:
        logger.error(f"Error extracting metrics: {str(e)}")
    
    return metrics


def _contains_non_year_number(text: str) -> bool:
    """Return whether text contains a quantitative magnitude rather than only a year."""
    without_urls = re.sub(r"https?://\S+", "", text)
    numbers = re.findall(
        r"(?<!\d)(?:\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(?!\d)",
        without_urls,
    )
    return any(
        not (number.isdigit() and len(number) == 4 and 1900 <= int(number) <= 2100)
        for number in numbers
    )


def enforce_quantification_disclosures(report_content: str) -> str:
    """Mark qualitative-only quantitative fields as unverified without inventing figures."""
    dimension_pattern = "|".join(
        re.escape(dimension) for dimension in DETAILED_ANALYSIS_DIMENSIONS
    )
    quantitative_pattern = "|".join(
        re.escape(dimension) for dimension in QUANTIFIED_DETAILED_ANALYSIS_DIMENSIONS
    )
    field_pattern = re.compile(
        rf"(?ms)(?P<prefix>^[ \t]*[-*][ \t]+\*\*(?P<label>{quantitative_pattern})(?::)?\*\*[ \t]*(?::)?[ \t]*)"
        rf"(?P<value>.*?)(?=^[ \t]*[-*][ \t]+\*\*(?:{dimension_pattern}|Competitive\ Threat\ Level)(?::)?\*\*|^###\s|^##\s|\Z)"
    )

    def add_disclosure(match: re.Match) -> str:
        value = match.group("value")
        normalized = value.lower()
        has_number = _contains_non_year_number(value)
        has_source = "source:" in normalized or "http://" in normalized or "https://" in normalized
        if has_number and not has_source:
            return (
                f'{match.group("prefix")}Unknown / not verified. '
                "The supplied magnitude had no inline trusted-source attribution.\n"
            )
        if "unknown / not verified" in normalized or has_number:
            return match.group(0)
        cleaned = value.strip()
        context = f" Qualitative context: {cleaned}" if cleaned else ""
        return f'{match.group("prefix")}Unknown / not verified.{context}\n'

    return field_pattern.sub(add_disclosure, report_content)


def competitor_report_issues(report_content: str, expected_competitors: int) -> List[str]:
    """Return structural problems that would make a generated report unusable in the UI."""
    content = (report_content or "").strip()
    issues = []
    if len(content) < 1200:
        issues.append("report is too short")

    required_sections = (
        "executive summary",
        "key findings",
        "detailed competitor analysis",
        "competitive comparison matrix",
        "strategic recommendations",
        "trusted source register",
    )
    lowered = content.lower()
    for section in required_sections:
        if section not in lowered:
            issues.append(f"missing {section} section")

    metrics = extract_key_metrics(content)
    if metrics["num_competitors"] < expected_competitors:
        issues.append(
            f"found {metrics['num_competitors']} competitor profiles; expected {expected_competitors}"
        )
    if not metrics["key_findings"]:
        issues.append("no parsable key findings")

    detailed = format_report_for_display(content)["detailed_analysis"]
    framework_labels = (
        "Industry Overview",
        "Competitor Landscape",
        *DETAILED_ANALYSIS_DIMENSIONS,
        "Competitive Strategy",
        "Actionable Recommendations",
    )
    missing_labels = [label for label in framework_labels if label.lower() not in detailed.lower()]
    if missing_labels:
        issues.append("detailed analysis missing framework fields: " + ", ".join(missing_labels))

    profile_matches = list(re.finditer(r"(?m)^###\s+(.+?)\s*$", detailed))
    incomplete_profiles = []
    unquantified_profiles = []
    for index, match in enumerate(profile_matches):
        profile_name = match.group(1).strip()
        profile_end = profile_matches[index + 1].start() if index + 1 < len(profile_matches) else len(detailed)
        profile_body = detailed[match.end():profile_end]
        profile_body_lower = profile_body.lower()
        missing_dimensions = [
            dimension for dimension in DETAILED_ANALYSIS_DIMENSIONS
            if dimension.lower() not in profile_body_lower
        ]
        if missing_dimensions:
            incomplete_profiles.append(f"{profile_name} ({', '.join(missing_dimensions)})")

        quantification_gaps = []
        for dimension in QUANTIFIED_DETAILED_ANALYSIS_DIMENSIONS:
            field_start = profile_body_lower.find(dimension.lower())
            if field_start < 0:
                continue
            later_starts = [
                profile_body_lower.find(other.lower(), field_start + len(dimension))
                for other in DETAILED_ANALYSIS_DIMENSIONS
            ]
            field_end_candidates = [position for position in later_starts if position >= 0]
            field_end = min(field_end_candidates) if field_end_candidates else len(profile_body)
            field_text = profile_body[field_start:field_end]
            normalized_field = field_text.lower()
            has_number = _contains_non_year_number(field_text)
            has_source = (
                "source:" in normalized_field
                or "http://" in normalized_field
                or "https://" in normalized_field
            )
            if "unknown / not verified" not in normalized_field and not has_number:
                quantification_gaps.append(dimension)
            elif has_number and not has_source:
                quantification_gaps.append(f"{dimension} [number lacks inline source]")
        if quantification_gaps:
            unquantified_profiles.append(f"{profile_name} ({', '.join(quantification_gaps)})")
    if incomplete_profiles:
        issues.append("incomplete competitor deep-dives: " + "; ".join(incomplete_profiles))
    if unquantified_profiles:
        issues.append(
            "quantitative fields require a non-year number or Unknown / not verified: "
            + "; ".join(unquantified_profiles)
        )
    return issues


def clean_text_for_export(text: str) -> str:
    """
    Clean text for export (remove special characters, etc.)
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    # Remove excessive whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    # Remove non-printable characters
    text = ''.join(char for char in text if char.isprintable() or char == '\n')
    
    return text.strip()


def generate_filename(company_name: str, extension: str = "pdf") -> str:
    """
    Generate a clean filename for exports
    
    Args:
        company_name: Company name
        extension: File extension (default: pdf)
        
    Returns:
        Clean filename
    """
    # Clean company name
    clean_name = re.sub(r'[^\w\s-]', '', company_name)
    clean_name = re.sub(r'[-\s]+', '_', clean_name)
    
    # Add timestamp
    timestamp = datetime.now().strftime('%Y%m%d')
    
    filename = f"competitor_analysis_{clean_name}_{timestamp}.{extension}"
    
    return filename
