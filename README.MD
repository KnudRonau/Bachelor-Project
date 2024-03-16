# Project Name

Software Project Artificial Consultation Bot

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)


## Introduction

This program enables users to download git repositories and ask an LLM questions regarding these through RAG. 

## Installation

1. Clone the repository: `git clone https://github.com/KnudRonau/DT002G_prototype`
2. Install the dependencies: `pip install langchain`, `pip install llama-cpp`, `pip install sentence-transformers`, `pip install gitpython`, `pip install chromadb`

## Usage

1. Run the program: run `GUI.py`.
2. Paste in a Git repository path, either as URL or local path (cannot be changed later).
3. Set desired temperature for you LLM (cannot be changed later).
4. Click `Select model` to choose your locally stored model to answer questions.
5. Ask a question regarding the repository in the text field and read the answer in the text area.

