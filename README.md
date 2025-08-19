# 📖 rss-digest

A lightweight Python tool that fetches RSS feeds, generates summaries with OpenAI, and produces a clean, ad-free HTML digest of articles in **night mode**.  
Perfect for distraction-free reading without ads, trackers, or clutter.

---

## ✨ Features
- 📰 Parse multiple RSS feeds
- 🧹 Extract full article text using [Mozilla Readability](https://github.com/mozilla/readability)
- 🌙 Dark mode HTML output (night-friendly colors)
- 🖼️ Resized and brightness-balanced images
- 🤖 Optional AI summaries powered by OpenAI
- 📌 Modal article view for easy navigation
- 🔖 Versioned with Git (stable releases tagged)

---

## 🚀 Installation

Clone the repository and set up a Python virtual environment:

```bash
git clone https://github.com/angelvalchev/rss-digest.git
cd rss-digest

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate   # On macOS/Linux
.venv\Scripts\activate      # On Windows

# Install dependencies
pip install -r requirements.txt
