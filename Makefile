# Variables
INPUT_CSV = data/raw_data/SPOTIFY_REVIEWS.csv
PROCESSED_CSV = data/processed/SPOTIFY_REVIEWS.csv
VECTORSTORE_PATH = vectorstore/
EMBEDDING_MODEL_NAME = sentence-transformers/all-MiniLM-L6-v2
DOCKER_IMAGE_NAME = spotify-review-chatbot
DOCKER_CONTAINER_NAME = chatbot-container

# Python environment setup (if using virtual environment)
VENV = .venv
PYTHON = $(VENV)/Scripts/python

.PHONY: all preprocess ingest deploy clean

all: preprocess ingest deploy

# Preprocess data
preprocess:
	@echo "Step 1: Preprocessing the data..."
	$(PYTHON) src/utils/preproces.py

# Ingest data to FAISS vectorstore
ingest:
	@echo "Step 2: Ingesting data into the vectorstore..."
	$(PYTHON) src/utils/ingest.py

# Deploy the chatbot application using Docker
deploy:
	@echo "Step 3: Building and deploying the Docker container..."
	# Build Docker image
	docker build -t $(DOCKER_IMAGE_NAME) .
	# Stop and remove the old container if it exists
	-docker stop $(DOCKER_CONTAINER_NAME)
	-docker rm $(DOCKER_CONTAINER_NAME)
	# Run the new container
	docker run -d --name $(DOCKER_CONTAINER_NAME) -p 8080:8080 $(DOCKER_IMAGE_NAME)

# Clean up build artifacts
clean:
	@echo "Cleaning up generated files..."
	rm -rf $(VECTORSTORE_PATH)
	rm -f $(PROCESSED_CSV)

# Create the virtual environment if not exists
$(VENV)/bin/activate:
	@echo "Creating virtual environment..."
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install -r requirements.txt

# Run the python scripts
$(PYTHON): $(VENV)/bin/activate
	$(PYTHON) --version
