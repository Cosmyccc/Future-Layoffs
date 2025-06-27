# Future-Layoffs

Future-Layoffs is a Python backend project that enables users to explore and ask questions about a GitHub code repository using the Groq language model. The application clones and indexes repositories, supports various file types, and leverages advanced language models to generate detailed, context-aware answers to user queries.

## Key Features
- Explore and query any GitHub repository interactively
- Clones and indexes the contents of a GitHub repository
- Supports various file types, including code, text, and Jupyter Notebook files
- Generates detailed answers to user queries based on the repository's contents
- Uses the Groq language model for generating responses
- Supports interactive conversation with the language model
- Presents top relevant documents for each question

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Cosmyccc/Future-Layoffs.git
   cd Future-Layoffs
   ```

2. **Install dependencies using Poetry:**
   ```bash
   poetry install
   ```

## Usage

1. **Activate the Poetry shell:**
   ```bash
   poetry shell
   ```

2. **Run the application:**
   ```bash
   python main.py
   ```

3. **Interact with the API:**
   - Use the provided API endpoints to submit a GitHub repository URL and ask questions about its contents.
   - The backend will clone and index the repository, then use the Groq language model to answer your queries.
   - Example endpoints:
     - `/api/v1/process/` - Submit a repository for processing
     - `/api/v1/query/` - Ask questions about the processed repository

## Project Structure
```
Future-Layoffs/
  ├── api/                # API endpoints (versioned)
  │   └── v1/
  │       ├── process/    # Processing endpoints
  │       └── query/      # Query endpoints
  ├── app/
  │   ├── controllers/    # Controller layer
  │   └── services/       # Business logic and utilities
  ├── core/
  │   ├── shared/         # Shared global state and utilities
  │   └── server.py       # Server entry point
  ├── main.py             # Main application entry
  ├── pyproject.toml      # Poetry configuration
  └── README.md           # Project documentation
```

## Contributing
Contributions are welcome! Please open issues or submit pull requests for improvements or bug fixes.

## License
This project is licensed under the MIT License.
