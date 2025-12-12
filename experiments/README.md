# 🤖 LLM Coding Experiments

A collection of one-shot prompts that generate self-contained HTML tools and experiments using Large Language Models.

## 🌐 Live Site

Visit the live site at: https://techczech.github.io/llmcodingexperiments

## 📁 Repository Structure

```
llmcodingexperiments/
├── .github/workflows/
│   └── pages.yml              # GitHub Pages deployment
├── experiments/               # Organized by LLM and version
│   ├── claude-3.5-sonnet/
│   │   ├── readability-analyzer.html
│   │   └── readability-analyzer-v2.html
│   ├── gpt-4/
│   │   └── readability-analyzer.html
│   ├── gemini-pro/
│   │   └── readability-analyzer.html
│   └── llama-3/
│       └── readability-analyzer.html
├── prompts/                   # The prompts used to generate tools
│   ├── readability-analyzer-prompt.md
│   └── readability-analyzer-v2-prompt.md
├── templates/
│   └── index.html            # Homepage template
├── scripts/
│   └── build.js             # Build script for GitHub Pages
├── experiments.json         # Metadata for all experiments
└── README.md
```

## 🚀 How to Add a New Experiment

1. **Organize by LLM**: Create or use existing subfolder in `experiments/` named after the LLM:
   ```
   experiments/gpt-4/
   experiments/claude-3.5-sonnet/
   experiments/gemini-pro/
   experiments/llama-3/
   ```

2. **Save your HTML tool**: Place the generated file in the appropriate LLM subfolder
3. **Save the prompt**: Save the exact prompt you used in the `prompts/` folder
4. **Update experiments.json**: Add your new experiment with the subfolder path:

```json
{
  "title": "Readability Analyzer (GPT-4)",
  "description": "GPT-4's implementation of the readability analysis tool",
  "filename": "gpt-4/readability-analyzer.html",
  "promptFile": "readability-analyzer-prompt.md",
  "date": "2025-05-23",
  "llm": "gpt-4",
  "version": "v1",
  "tags": ["readability", "text-analysis", "benchmark"]
}
```

5. **Commit and push**: The site will automatically rebuild and deploy via GitHub Actions

## 🧪 LLM Benchmark Prompt

**Standard Test Prompt:** `"Make a detailed, visually interesting readability analysis tool."`

This prompt serves as an excellent benchmark for comparing different LLMs because it requires multiple capabilities:

### 📚 **Domain Knowledge**
- Understanding of readability metrics (Flesch-Kincaid, Coleman-Liau, etc.)
- Knowledge of text analysis principles
- Awareness of educational/accessibility concepts

### 🎨 **Design Planning**
- UI/UX design decisions
- Visual hierarchy and layout
- Color theory and user experience

### 💻 **Technical Implementation**
- HTML/CSS/JavaScript coding
- Algorithm implementation (syllable counting, readability formulas)
- Real-time interaction and responsiveness

### 🔍 **Model Performance Indicators**
- **Advanced models**: Generate comprehensive tools with multiple metrics, professional design, and complex features
- **Basic models**: May only create simple text analysis or basic functionality
- **Mid-tier models**: Often handle one or two aspects well but miss others

This makes it ideal for testing and comparing the capabilities of different LLMs across knowledge domains, planning abilities, and coding skills all in a single prompt.

## 🔄 Version Management

You can create multiple versions:
- **Different LLMs**: `gpt-4/tool.html`, `claude-3.5-sonnet/tool.html`
- **Different versions**: `claude-3.5-sonnet/tool-v1.html`, `claude-3.5-sonnet/tool-v2.html`
- **Iterative improvements**: Use "make it better" prompts for v2, v3, etc.

## 📝 Experiments.json Format

Each experiment entry should include:
- `title`: Display name including LLM identifier
- `description`: Brief description of implementation
- `filename`: Path including subfolder (e.g., "gpt-4/tool.html")
- `promptFile`: Name of the prompt file in prompts/
- `date`: Date created (YYYY-MM-DD format)
- `llm`: LLM identifier (e.g., "gpt-4", "claude-3.5-sonnet")
- `version`: Version identifier (e.g., "v1", "v2")
- `tags`: Array of tags for categorization

## 🤝 Contributing

Feel free to:
- Test the same prompt with different LLMs
- Add iterative improvements with "make it better"
- Create new benchmark prompts
- Enhance the comparison features

## 📄 License

MIT License - feel free to use this structure for your own projects!