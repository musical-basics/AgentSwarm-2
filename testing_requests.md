# Swarm Architect - Test Scenarios

Use these prompts in the **Swarm Input** field to verify that the Architect is planning correctly, respecting budgets, and following dependency logic.

## 1. Single Node (Simple)
**Prompt:** "Research the current weather in Tokyo and tell me if I need an umbrella."
- **Expected Result:** 1 Researcher node using a Cheap model.

## 2. Parallel Processing
**Prompt:** "Compare the 2024 quarterly earnings of Apple, Nvidia, and Microsoft. Create a summary table."
- **Expected Result:** 3 Parallel Researcher nodes followed by 1 Analyst node.

## 3. High-Quality Coding (Premium)
**Prompt:** "Write a robust Python script to scrape news from HackerNews and save it to a PostgreSQL database. Include error handling."
- **Expected Result:** 1 Coder node using a **High-tier** model (e.g., GPT-4o or Claude 3.5 Sonnet).

## 4. Budget Constraint (Low)
**Prompt:** "Write a short story about a robot learning to paint."
**Budget:** $0.05
- **Expected Result:** 1 Coder/Analyst node restricted to a **Cheap** model (e.g., Gemini Flash or GPT-4o-mini).

## 5. Multi-Stage Pipeline
**Prompt:** "Research the latest trends in renewable energy, analyze the most promising technology, and write a blog post about it."
- **Expected Result:** Researcher -> Analyst -> Coder/Writer (Sequential).

## 6. Critical Approval (HITL Gateway)
**Prompt:** "Deploy a critical security patch to the production server and verify the checksum."
- **Expected Result:** Architect should flag the 'deployment' or 'patch' node as `approval_required: true`. (Look for the gold banner).

## 7. Quality Assurance (Self-Correction)
**Prompt:** "Write a complex regex for email validation and have a specialized agent verify its edge cases."
- **Expected Result:** 1 Coder node followed by 1 **QA** node.

## 8. Financial Analysis (Math Heavy)
**Prompt:** "Analyze my 'stocks.csv' file, calculate the RSI for the last 14 days, and suggest which ones to sell."
- **Expected Result:** 1 Researcher (to read file) -> 1 Analyst (for RSI math).

## 9. Repair Mode (Simulated Failure)
**Prompt:** "Generate a broken JSON file and then fix it using a validation step."
- **Expected Result:** If the first node fails or the QA node flags it, you should see the **"Self-Correction Triggered"** message in chat.

## 10. Large Context Research
**Prompt:** "Summarize the entire 200-page Logic_Architect_PRD.md and list every technical requirement for the DAG executor."
- **Expected Result:** High-tier model with large context length (selected via `models_tiering.py`).
