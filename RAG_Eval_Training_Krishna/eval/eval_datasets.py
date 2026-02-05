"""eval/datasets.py

Evaluation dataset = questions + (optional) ground truth answers.

INTERVIEW TIP:
- You can evaluate without ground truth (faithfulness, answer relevance),
  but context recall/precision needs ground truth.
- In real enterprise you maintain a small "golden set" per domain.
"""

from __future__ import annotations
from typing import List, Dict

# A small "golden" set for training.
# You can add more questions as you learn the policy.
EVAL_QUESTIONS: List[Dict[str, str]] = [
    {
        "question": "What are the standard working hours?",
        "ground_truth": "Staff members are required to work 9:00 AM to 5:30 PM Monday to Friday, with one hour lunch, totaling 40 hours per week.",
    },
    {
        "question": "How much annual leave is a full-time employee entitled to?",
        "ground_truth": "Full-time staff are entitled to 24 days of annual leave per annum (accruing at 2 days per month), with the leave year running Jan 1 to Dec 31.",
    },
    {
        "question": "What is the probationary period for new staff?",
        "ground_truth": "For contracts exceeding 12 months, the probationary period is six months and may be extended up to a further six months by agreement.",
    },
    {
        "question": "Under what conditions can a salary advance be given and what is the maximum amount?",
        "ground_truth": "Salary advances may be considered for serious illness, legitimate emergencies, or exceptional compelling circumstances with CEO authorization, and should not exceed one month’s net earnings (recovered from subsequent salary).", 
    },
    {
        "question": "What are the eligibility rules and limitations for relocation benefits?",
        "ground_truth": "Eligible staff relocating on appointment or change of duty station may have household goods and personal effects shipped by the most economical means, subject to maximum allowances under the relevant scheme rules and limits.",
    },
    {
        "question": "What are the maternity leave entitlements and pay rules?",
        "ground_truth": "The policy describes maternity-related leave after birth taken continuously or in separate periods within a year; specific entitlement/pay details must be confirmed in the policy section on maternity leave if present.",
    },
    {
        "question": "If an employee is found guilty of gross misconduct after suspension, what rights and financial consequences apply?",
        "ground_truth": "Staff are suspended with full pay during investigation and must be given a chance to explain; on summary dismissal they are not entitled to pay in lieu of notice or severance pay.",
    },
    {
        "question": "Can an employee do paid consultancy on the side? What happens if they do it without permission?",
        "ground_truth": "Staff are not allowed to undertake paid consultancies or other paid employment during their employment (including evenings/weekends/leave); doing so may lead to disciplinary action as unprofessional conduct.",
    },
    {
        "question": "Does the organization offer stock options or equity participation?",
        "ground_truth": "The HR policy does not mention a stock/equity plan.",
    },
    {
        "question": "What is the policy on working from home?",
        "ground_truth": "With consent, home is treated as a work location under occupational safety rules; health and safety policies apply, the organization may inspect the home-work environment, and issues must be resolved to continue the arrangement.",
    },
]
