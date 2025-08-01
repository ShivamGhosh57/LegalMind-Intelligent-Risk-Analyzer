# Stage 1: Use an official Python runtime as a parent image
# Using a slim version to keep the image size smaller
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Upgrade pip to ensure we are using the latest version
RUN python -m pip install --upgrade pip

# Copy all project files into the container first.
# This ensures setup.py is available when pip runs.
COPY . .

# Install the Python dependencies from requirements.txt
# And add a verification step to ensure streamlit is installed correctly.
# The build will fail here if streamlit is not found.
RUN pip install --no-cache-dir -r requirements.txt && \
    python -m streamlit --version

# Expose the port that the Flask app will run on
EXPOSE 5001

# Define the command to run your application
# This command will be executed when the container starts
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.port=5001", "--server.address=0.0.0.0"]
