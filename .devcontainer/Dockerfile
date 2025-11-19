# .devcontainer/Dockerfile
FROM mcr.microsoft.com/devcontainers/python:3.11

# Set working directory
WORKDIR /workspace

# Copy Python requirements and install dependencies
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the server application
COPY server.py /workspace/server.py

# Create directories for workflows
# - /workspace/proposed_workflows: for your input workflow files
# - /workspace/workflow_set: for processed/stored workflows
RUN mkdir -p /workspace/proposed_workflows \
    && mkdir -p /workspace/workflow_set

# Expose Flask port
EXPOSE 8000

# Default to bash for development
CMD ["bash"]
