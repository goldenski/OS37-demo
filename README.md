# OS37 Healthcare Analytics Demo

A privacy-native, stateless healthcare analytics platform with client-side redaction (Pyodide WASM), OpenAI GPT-4 analytics agent, and interactive dashboards.

## Features
- **Client-side privacy redaction**: No identifiers or raw data leave your browser.
- **AI-powered analytics**: GPT-4 agent with function calling for healthcare insights.
- **Stateless architecture**: No server-side persistence, zero-knowledge proof interface.
- **Interactive dashboards**: Descriptive, comparative, survival, and predictive analytics with Plotly.
- **PDF and report export**: Download results with OS37 branding, no server storage.
- **Production ready**: Mobile responsive, robust error handling, and Streamlit Cloud deployable.

## Quick Start
1. Clone this repo and install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2. Add your OpenAI key to `.streamlit/secrets.toml`:
    ```toml
    OPENAI_API_KEY = "your-key-here"
    ```
3. (Optional) Set your theme in `.streamlit/config.toml`:
    ```toml
    [theme]
    primaryColor = "#3b82f6"
    backgroundColor = "#0f172a"
    secondaryBackgroundColor = "#1e293b"
    textColor = "#f1f5f9"
    ```
4. Run the app:
    ```bash
    streamlit run app.py
    ```

## Privacy & Security
- All processing is done in-browser using WASM (Pyodide).
- No patient identifiers or raw data are sent to the server.
- Zero file system persistence and stateless by design.
- Zero-knowledge cryptographic proof interface included.

## Demo Video
[![Watch Demo](https://img.youtube.com/vi/dQw4w9WgXcQ/0.jpg)](https://github.com/os37-org/demo-assets/releases/download/v1.0/os37-demo.mp4)

## Architecture
- **Frontend**: Streamlit, Plotly, Pyodide WASM, custom JS
- **Backend**: Stateless, OpenAI GPT-4 agent (function calling)
- **Privacy**: Client-side redaction, hashing, and proof

## Screenshots
![Dashboard](assets/demo_dashboard.png)
![Redaction](assets/demo_redaction.png)

## License
See [LICENSE](LICENSE).

## Contact
- Project: [github.com/os37-org/os37-demo](https://github.com/os37-org/os37-demo)
- Email: privacy@os37.org
- Issues: [GitHub Issues](https://github.com/os37-org/os37-demo/issues)
