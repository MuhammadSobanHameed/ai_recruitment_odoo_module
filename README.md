# 🧠 Odoo AI Recruitment Module  
### Intelligent CV Parsing, Matching, and Recruitment Automation

This module provides an **AI-augmented recruitment workflow** inside Odoo.  
It enhances the hiring process by adding AI-assisted features without exposing internal business logic.

---

## 🚀 Features (High-Level Only)

- ✔ AI-assisted CV parsing  
- ✔ Extracts candidate details automatically  
- ✔ Generates job descriptions using LLMs  
- ✔ Matches candidates to job positions  
- ✔ Adds an “AI Suggestion” panel in applicant form  
- ✔ One-click workflow automation  
- ✔ API endpoints to connect external AI engines  

> ⚠️ This repository contains only **module structure + placeholders**.  
> Your private AI logic should be placed inside the designated sections in the Python files.

---

## 📂 Module Structure

- `models/`  
  Contains logic placeholders for CV parsing, matching, and AI helper functions.

- `controllers/`  
  Optional REST API layer for external AI engines.

- `views/`  
  Minimal UI enhancements including buttons and result panels.

- `security/`  
  Access rights for AI models.

---

## 🛠 Installation

1. Copy the folder `ai_recruitment` into your Odoo `addons` directory.
2. Activate developer mode.
3. Update Apps List.
4. Search for **AI Recruitment** and install.

---

## 🧩 Configuration

In Settings → AI Integration:

- Add your API key  
- Choose provider (OpenAI, LLaMA, Gemini, etc.)
- Enable/disable AI Job Matching feature

---

## 🚧 Add Your Private AI Logic

Inside:

