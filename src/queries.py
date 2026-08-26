"""
Shared query definitions used by both main.py and experiments/run_experiments.py.

Add or remove queries here — both files stay in sync automatically.
"""

TEST_CASES = [
    {
        "label": "Q1 Direct",
        "query": "What is the policy on international travel?",
        "expect_section": "International Travel Policy",
        "expect_result": True,
    },
    {
        "label": "Q2 Standard",
        "query": "What are the remote work options available?",
        "expect_section": "Remote Work Policy",
        "expect_result": True,
    },
    {
        "label": "Q3 Multi-section",
        "query": "What products does the company offer?",
        "expect_section": "Products and Services",
        "expect_result": True,
    },
    {
        "label": "Q4 Cross-section",
        "query": "What is the meal allowance for domestic and international travel?",
        "expect_section": "Domestic Travel Policy",
        "expect_result": True,
    },
    {
        "label": "Q5 Specific detail",
        "query": "What certifications does SecureID have?",
        "expect_section": "Products and Services",
        "expect_result": True,
    },
    {
        "label": "Q6 Out-of-scope",
        "query": "What is the company's policy on cryptocurrency investment?",
        "expect_section": None,
        "expect_result": False,
    },
]

# Plain query strings for main.py
QUERIES = [tc["query"] for tc in TEST_CASES]
