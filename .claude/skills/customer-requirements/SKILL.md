---
name: customer-requirements
description: "Read customer requirements from an Excel spreadsheet, validate understanding with the user, and turn them into a prioritised development plan that gets executed. Use this skill when the user explicitly asks to process customer requirements, work from a requirements spreadsheet, or plan features from an Excel file."
---

# Customer Requirements to Development Plan

Turn a structured Excel requirements spreadsheet into validated, prioritised code changes.

## Spreadsheet Format

The Excel file has two tabs:

### Customer Requirements tab
Each row is one requirement with these columns:
- **Ref** — reference number (not priority or sequence)
- **Priority** — `Must Do` (MVP) | `Should Do` (phase 2) | `Could Do` (revisit later) | `Won't Do` (explicitly out of scope)
- **Scope** — `New` (incremental feature) or `Revision` (change to existing functionality)
- **Status** — `Open` (to implement) or `Done` (context only)
- **Feature Category** — groups related features
- **Feature Description** — what to deliver
- **Motivation** — why, and how the user will use it
- **Exit Criteria** — minimum conditions for the requirement to be considered done

### Glossary tab
- **Term** — keyword exactly as it appears in the spreadsheet
- **Meaning** — its definition

## Process

### Step 1: Validate understanding

Read the entire spreadsheet (both tabs), then do three things before any planning or coding:

**a. Named Entity Recognition**
Scan all text fields for named entities — applications, packages, protocols, proper names. Check each against: your own knowledge, web search, and the Glossary tab. Ask the user to clarify any that are ambiguous or undefined.

**b. Exit Criteria audit**
Review every Exit Criteria cell. Each must be a declarative, verifiable outcome actionable through code. Flag any that contain questions, human-only actions (e.g. "ask the customer for feedback"), or vague/missing outcomes. Ask the user to clarify before proceeding.

**c. Feature summaries**
Produce and present to the user:
1. A brief overview of all features and objectives (5–7 sentences)
2. A per-Feature-Category summary (3–5 sentences each)

Once confirmed, save these summaries to memory and consult the relevant one whenever starting work on a requirement in that category.

### Step 2: Plan and execute

- Exclude `Done` requirements — they exist for context only
- For `Revision`-scoped requirements, read the existing code and deliver only the delta
- Prioritise `Must Do` first, then `Should Do`, respecting dependencies between requirements
- Enter planning mode, map delivery steps to specific file changes, and share with the user
- Once confirmed, implement each step and verify Exit Criteria are met before moving on
