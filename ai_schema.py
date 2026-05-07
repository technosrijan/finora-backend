from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union

# --- MAP PHASE SCHEMA ---
class MapExtractionSchema(BaseModel):
    extracted_metrics: List[Dict[str, str]] = Field(description="List of raw numerical metrics or KPIs found in this section. e.g. {'name': 'Revenue', 'value': '$10M', 'trend': '+5% YoY', 'context': 'Driven by subscription growth'}")
    extracted_risks: List[str] = Field(description="Any risks, litigations, or threats mentioned in this section.")
    extracted_strategies: List[str] = Field(description="Any strategic initiatives, business models, or future plans mentioned.")
    extracted_ratios: List[Dict[str, str]] = Field(default_factory=list, description="Financial ratios found: e.g. {'name': 'P/E Ratio', 'value': '25.3x', 'context': 'Industry avg 20x'}")
    extracted_revenue_segments: List[Dict[str, str]] = Field(default_factory=list, description="Revenue breakdown by segment/geography: e.g. {'segment': 'Cloud Services', 'value': '45%', 'amount': '$4.5B'}")

# --- REDUCE PHASE SCHEMA (Final Dashboard) ---
class DynamicMetric(BaseModel):
    name: str = Field(description="Name of the metric, e.g., 'Monthly Active Users' or 'Inventory Turnover'")
    value: str = Field(description="The value of the metric, e.g., '150 million' or '4.5x'")
    trend: str = Field(description="The trend or year-over-year change, e.g., '+5% YoY' or 'Not disclosed'")
    context: str = Field(description="A brief 1-sentence explanation of what drove this metric.")

class KeyRatio(BaseModel):
    name: str = Field(description="Name of the financial ratio, e.g. 'P/E Ratio', 'ROE', 'Debt-to-Equity', 'Current Ratio', 'ROCE'")
    value: str = Field(description="The value, e.g. '25.3x', '18.5%', '0.65'")
    assessment: str = Field(description="One of: 'strong', 'moderate', 'weak', 'neutral'")
    context: str = Field(description="Brief explanation, e.g. 'Above industry average of 15%'")

class RevenueSegment(BaseModel):
    segment: str = Field(description="Name of the revenue segment, e.g. 'Cloud Services', 'North America', 'Subscriptions'")
    value: float = Field(description="Percentage share of total revenue as a number 0-100, e.g. 45.0")
    amount: str = Field(description="Absolute revenue amount for this segment if available, e.g. '$4.5B'. Use 'N/A' if not disclosed.")

class DynamicChartDataPoint(BaseModel):
    label: str = Field(description="The label for the X-axis (e.g., '2021', 'Q3', 'North America')")
    value: float = Field(description="The numerical value for this point")

class DynamicChart(BaseModel):
    title: str = Field(description="Title of the chart, e.g., 'Revenue vs Costs (2020-2023)'")
    chart_type: str = Field(description="Must be exactly one of: 'bar', 'line', 'pie', 'area'")
    x_axis_label: str = Field(description="Label for the X axis")
    y_axis_label: str = Field(description="Label for the Y axis")
    data_points: List[DynamicChartDataPoint] = Field(description="List of data points to plot")

class ReportInsights(BaseModel):
    company_name: str = Field(description="The name of the company or entity this report is about.")
    reporting_period: str = Field(description="The financial period or year this report covers (e.g. 'FY24' or 'Q3 2023').")
    sector: str = Field(default="General", description="The industry sector, e.g. 'Technology', 'Banking & Finance', 'Healthcare', 'Energy', 'Consumer Goods', 'Industrial', 'Real Estate'.")
    financial_health_score: float = Field(default=5.0, description="Overall financial health rating from 1.0 (critical) to 10.0 (excellent). Base this on profitability, debt levels, growth trajectory, and cash flow quality.")
    sentiment_score: float = Field(default=0.0, description="Overall document sentiment from -1.0 (very negative/bearish) to 1.0 (very positive/bullish). Base on management tone, outlook statements, and risk disclosures.")
    executive_summary: str = Field(description="A comprehensive executive summary of the entire document in 4-6 sentences.")
    key_metrics: List[DynamicMetric] = Field(description="A list of the 10 to 20 most critical metrics extracted from the document. Prioritize: Revenue, Net Profit, EBITDA, Operating Margin, EPS, Free Cash Flow, then sector-specific KPIs.")
    key_ratios: List[KeyRatio] = Field(default_factory=list, description="5 to 10 key financial ratios: P/E, P/B, ROE, ROCE, Debt-to-Equity, Current Ratio, Interest Coverage, Dividend Yield, etc.")
    revenue_breakdown: List[RevenueSegment] = Field(default_factory=list, description="Revenue breakdown by business segment or geography. 3-8 segments. Values should sum to approximately 100.")
    generated_charts: List[DynamicChart] = Field(description="3 to 6 comprehensive charts generated from the tabular data and numerical metrics. Include at least one 'bar', one 'line', and optionally 'pie' or 'area' charts.")
    risk_analysis: List[str] = Field(description="A list of the primary risks, threats, or contingent liabilities.")
    strategic_initiatives: List[str] = Field(description="A list of core strategic plans, business models, or CapEx initiatives.")

class ComparativeMetric(BaseModel):
    metric_name: str = Field(description="Name of the metric, e.g., 'Revenue YoY Growth' or 'Operating Margin'")
    winner: str = Field(description="Name of the company that performs best in this metric, or 'Tie'")
    rationale: str = Field(description="Brief explanation of why they won and the significance")

class CompanyStrength(BaseModel):
    company_name: str = Field(description="Name of the company")
    key_strengths: List[str] = Field(description="2-3 key competitive advantages based on the reports")
    key_weaknesses: List[str] = Field(description="1-2 primary weaknesses or risks")

class AIComparison(BaseModel):
    executive_summary: str = Field(description="A high-level synthesis comparing the companies, their market positions, and overall financial health.")
    market_leader: str = Field(description="The company that appears to be the overall strongest, based on the provided metrics.")
    comparative_metrics: List[ComparativeMetric] = Field(description="Detailed comparison of 4-6 key metrics across the companies.")
    company_profiles: List[CompanyStrength] = Field(description="Strengths and weaknesses for each company compared.")
