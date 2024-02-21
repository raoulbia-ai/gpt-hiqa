# Use the appropriate base image
FROM python:3.8

# Set the working directory in the container
WORKDIR /app

# Copy the application files and the .env file into the container
COPY . /app
COPY .env /app/.env


# Copy the persist directory into the container
COPY data/persist /app/data/persist

# Install any necessary dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Start the Streamlit application
CMD ["streamlit", "run", "src/multi_doc\-agents.py", "--server.port=80"]
