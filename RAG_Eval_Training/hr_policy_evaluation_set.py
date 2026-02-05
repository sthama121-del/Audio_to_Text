# =============================================================================
# gesci_evaluation_set.py — Ground-Truth Evaluation Dataset
# Source: HR_POLICY.pdf
# =============================================================================
#
# 10 questions across 4 difficulty tiers:
#   Tier 1 — Simple Factual Lookup       (Q1–Q3)
#   Tier 2 — Multi-detail Extraction     (Q4–Q6)
#   Tier 3 — Cross-Section Synthesis     (Q7–Q8)
#   Tier 4 — Out-of-Scope Detection      (Q9–Q10)
#
# Each entry:
#   question      – what the user asks the RAG bot
#   ground_truth  – the correct, complete answer grounded in the doc
#   source_section – which section(s) the answer lives in (for debugging retrieval)
#   difficulty    – the tier label
# =============================================================================

evaluation_set = [

    # ──────────────────────────────────────────────────────────────────────
    # TIER 1 — Simple Factual Lookup
    # These test basic retrieval + faithful summarisation.
    # A working pipeline should nail all three.
    # ──────────────────────────────────────────────────────────────────────

    {
        "question": "What are the standard working hours at GESCI?",
        "ground_truth": (
            "The standard working hours are 9:00 AM to 5:30 PM, Monday to Friday, "
            "with a one-hour lunch break. The official weekly working hours total 40 hours."
        ),
        "source_section": "Section 5 – Attendance and Leave (5.1 Hours of Work)",
        "difficulty": "Tier 1 – Simple Factual",
    },

    {
        "question": "How much annual leave is a full-time GESCI employee entitled to?",
        "ground_truth": (
            "The standard annual leave entitlement for full-time staff is 24 days per annum. "
            "Leave accrues at the rate of 2 days per month. The annual leave year runs from "
            "1 January to 31 December."
        ),
        "source_section": "Section 5 – Attendance and Leave (5.2.1 Annual Leave Entitlement)",
        "difficulty": "Tier 1 – Simple Factual",
    },

    {
        "question": "What is the probationary period for new GESCI staff?",
        "ground_truth": (
            "The probationary period is six months for all staff on contracts exceeding 12 months. "
            "It may be extended by a further period (up to a maximum of six additional months) "
            "with the agreement of the staff member, if performance is below standard or for disciplinary reasons."
        ),
        "source_section": "Section 3 – Recruitment and Appointment (3.6 Probation)",
        "difficulty": "Tier 1 – Simple Factual",
    },

    # ──────────────────────────────────────────────────────────────────────
    # TIER 2 — Multi-Detail Extraction
    # These require the pipeline to retrieve a single section but pull
    # several specific data points from it. Good test of context recall.
    # ──────────────────────────────────────────────────────────────────────

    {
        "question": (
            "Under what specific conditions can a salary advance be given to a GESCI employee, "
            "and what is the maximum amount?"
        ),
        "ground_truth": (
            "A salary advance may be given in the following situations: (1) when an employee has "
            "not received regular pay through no fault of their own; (2) upon separation from service "
            "when final settlement cannot be made, up to 80% of estimated final net payments; "
            "(3) upon an official change of duty station, up to 50% of one month's salary; "
            "(4) in cases of serious illness or legitimate emergencies. The CEO may also authorise "
            "an advance for any other reason if the request is supported by detailed written justification. "
            "In all cases, the advance shall not exceed one month's net earnings and must be recovered "
            "from the staff member's next salary."
        ),
        "source_section": "Section 4 – Remuneration and Benefits (4.3.1 Salary Advances)",
        "difficulty": "Tier 2 – Multi-Detail Extraction",
    },

    {
        "question": (
            "What are the eligibility rules and limitations for relocation benefits at GESCI?"
        ),
        "ground_truth": (
            "Relocation benefits are available to staff appointed to posts subject to international "
            "recruitment. However, a staff member who has lived in the duty station area for two years "
            "or more prior to appointment is not eligible. Staff officially transferred between duty "
            "stations are also entitled on first move and on termination. Eligible staff may receive "
            "assistance towards temporary accommodation for up to two weeks. Shipment allowances are: "
            "4,890 kg for staff without dependents, and 8,150 kg for staff with dependents. "
            "Removal and shipment benefits on termination must be used within 3 months of termination "
            "(the CEO may extend this to a maximum of 6 months), and only if the new employer is not "
            "covering relocation costs."
        ),
        "source_section": "Section 4 – Remuneration and Benefits (4.4.4 Relocation Benefits)",
        "difficulty": "Tier 2 – Multi-Detail Extraction",
    },

    {
        "question": (
            "What are the maternity leave entitlements for a female GESCI staff member, "
            "and what are the rules around pay and additional leave?"
        ),
        "ground_truth": (
            "A female staff member is entitled to a total of 16 weeks of maternity leave. "
            "Pre-delivery leave begins six weeks before the anticipated birth date (or may be shortened "
            "to two weeks if a medical certificate confirms fitness to continue working). Post-delivery "
            "leave covers the remainder of the 16 weeks, with a minimum of 10 weeks post-delivery. "
            "The entire 16 weeks are paid at full pay. Annual leave continues to accrue during "
            "maternity leave. A nursing staff member is also entitled to one hour of paid leave per day "
            "for two months after returning. Additionally, a staff member may request up to 16 weeks "
            "of unpaid leave beyond the standard entitlement, at the CEO's discretion."
        ),
        "source_section": "Section 5 – Attendance and Leave (5.3.3 Parental Leave – Maternity)",
        "difficulty": "Tier 2 – Multi-Detail Extraction",
    },

    # ──────────────────────────────────────────────────────────────────────
    # TIER 3 — Cross-Section Synthesis
    # These require retrieving and combining information from two or more
    # different sections. This is where weak chunking/retrieval breaks down.
    # ──────────────────────────────────────────────────────────────────────

    {
        "question": (
            "If a GESCI employee is found guilty of gross misconduct after being suspended, "
            "what are their rights during the process and what financial consequences do they face upon dismissal?"
        ),
        "ground_truth": (
            "During the process, the employee has the right to be notified in writing of allegations, "
            "the right to state their case and be represented by a colleague, the right to cross-examine "
            "witnesses, and the right to call their own witnesses. They may appeal to the CEO within "
            "7 days, and further appeal to the Board within 7 days of the CEO's decision; the Board's "
            "decision is final. During suspension with pay pending investigation, benefits and entitlements "
            "are protected. However, upon summary dismissal for gross misconduct, the employee is NOT "
            "entitled to pay in lieu of notice, nor to any severance pay."
        ),
        "source_section": "Section 9 – Disciplinary (9.3 Procedures, 9.3.5 Suspension, 9.3.6 Summary Dismissal) + Section 10 – Separation (9.4.1)",
        "difficulty": "Tier 3 – Cross-Section Synthesis",
    },

    {
        "question": (
            "An employee wants to take on a paid consultancy on the side while still working at GESCI. "
            "What does the policy say about this, and what could happen if they do it without permission?"
        ),
        "ground_truth": (
            "GESCI staff are not allowed to undertake paid consultancies or any other form of paid "
            "employment during their employment without the CEO's express authorisation, which may be "
            "withdrawn at any time. This restriction applies even during evenings, weekends, or while on leave. "
            "If payment is received in appreciation of work done, it must be declared and reverted to GESCI. "
            "Failure to observe the conflict-of-interest policy is considered a disciplinary offence subject "
            "to GESCI's disciplinary procedures. Furthermore, undertaking outside employment without the "
            "consent of the CEO is explicitly listed as a gross misconduct offence, which can result in "
            "immediate summary dismissal."
        ),
        "source_section": "Section 2 – Duties and Obligations (2.2.1 Outside Employment, 2.2.2 Conflict of Interest) + Section 9 – Gross Misconduct (xviii)",
        "difficulty": "Tier 3 – Cross-Section Synthesis",
    },

    # ──────────────────────────────────────────────────────────────────────
    # TIER 4 — Out-of-Scope Detection
    # These questions have NO answer in the document. A well-built RAG
    # pipeline must recognise this and refuse to fabricate an answer.
    # This is the single most important test for faithfulness.
    # ──────────────────────────────────────────────────────────────────────

    {
        "question": "Does GESCI offer a stock option or equity participation plan for its employees?",
        "ground_truth": (
            "The GESCI HR policy document does not contain any information about stock options or "
            "equity participation plans. This question cannot be answered from the available document."
        ),
        "source_section": "N/A – Not covered in document",
        "difficulty": "Tier 4 – Out-of-Scope Detection",
    },

    {
        "question": "What is GESCI's policy on remote work or working from home arrangements?",
        "ground_truth": (
            "The document does not contain a dedicated remote work or work-from-home policy. "
            "The only related reference is in the Safety, Health and Welfare section (Section 8.6), "
            "which states that if a staff member works from home, their home is regarded as a "
            "location of work under the Occupational Safety and Health Act (OSHA), and GESCI "
            "reserves the right to inspect the home-working environment. No eligibility criteria, "
            "approval process, or day-count limits for remote work are defined in this document."
        ),
        "source_section": "Section 8 – Safety (8.6 Working from Home) – partial only",
        "difficulty": "Tier 4 – Out-of-Scope Detection (partial)",
    },

]


# ---------------------------------------------------------------------------
# Quick sanity check — run this file directly to print the set
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Total questions: {len(evaluation_set)}\n")
    for i, item in enumerate(evaluation_set, 1):
        print(f"Q{i} [{item['difficulty']}]")
        print(f"   Question : {item['question'][:90]}...")
        print(f"   Section  : {item['source_section']}")
        print()
