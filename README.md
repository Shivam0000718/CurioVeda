# CurioVeda
# CurioVeda

CurioVeda is an AI-powered web scraping and question-answering tool. This application allows users to input a website's URL, scrape the data present on the HTML page, and query the content using an intuitive interface. Built using Python and Flask, CurioVeda showcases the potential of combining web scraping and natural language processing (NLP) to enhance content accessibility and usability.

---

## Features

- **Web Scraping**: Extracts data from the HTML pages of websites using the provided URLs.
- **Learning from Content**: Processes and structures the extracted data for efficient querying.
- **Intelligent Q&A**: Enables users to ask questions related to the content on the page, providing accurate and contextual answers.
- **User-Friendly Interface**: Built with Flask to ensure a seamless user experience.

---

## Tech Stack

- **Programming Language**: Python
- **Web Framework**: Flask
- **Web Scraping Library**: BeautifulSoup
- **AI/NLP Techniques**: Pre-trained models and custom algorithms for content processing and question-answering.

---

## Installation and Setup

Follow the steps below to set up CurioVeda on your local machine:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Shivam0000718/curioveda.git
   cd CurioVeda
   ```

2. **Create and Activate a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**:
   ```bash
   python app.py
   ```

5. **Access the Web Application**:
   Open your web browser and navigate to `http://127.0.0.1:5000/`.

---

## Usage

1. Enter the URL of the website whose content you want to scrape.
2. The application will process the webpage and extract the HTML content.
3. Ask questions related to the content, and CurioVeda will provide intelligent and accurate answers.

---

## Screenshots

![Screenshot 2025-01-12 090340](https://github.com/user-attachments/assets/004d66c1-bd17-47a6-9c6e-88870199480f)
![Screenshot 2025-01-12 090405](https://github.com/user-attachments/assets/32f57d0c-4c91-461a-8a30-4db92d5d0446)


---

## Future Enhancements

- **Support for Dynamic Websites**: Extend functionality to scrape content from dynamic pages (e.g., JavaScript-rendered content).
- **Content Summarization**: Provide summarized insights from the scraped data.
- **Enhanced Q&A**: Integrate more advanced NLP models for improved answer quality.
- **Export Options**: Allow users to download the scraped data in various formats (e.g., JSON, CSV).

---

## Contribution

We welcome contributions to improve CurioVeda! To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a pull request.


---

## Contact

For any queries or support, please reach out to:

- **Name**: Shivam Mishra
- **Email**: shivam90above@gmail.com

---

## Acknowledgments

- Libraries like [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/) and frameworks like [Flask](https://flask.palletsprojects.com/) for making development seamless.
- Inspiration from cutting-edge web scraping and NLP tools.
