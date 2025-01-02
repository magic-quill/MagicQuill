# syntax=docker/dockerfile:1.4

FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Install OpenGL and other dependencies
RUN apt-config dump | grep -we Recommends -e Suggests | sed s/1/0/ | tee /etc/apt/apt.conf.d/999norecommend && \
    apt-get update && apt-get upgrade -y && \
    apt-get install -y libglib2.0-0 libgl1 wget vim && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Create a non-root user
ARG APP_USER=quill
RUN useradd -m ${APP_USER} -s /bin/bash
USER ${APP_USER}
WORKDIR /home/${APP_USER}
ENV PATH="/home/${APP_USER}/.local/bin:${PATH}"

# Copy the Magic Quill application and installation files
ADD --chown=${APP_USER}:${APP_USER} MagicQuill /home/${APP_USER}/MagicQuill/MagicQuill
ADD --chown=${APP_USER}:${APP_USER} --chmod=750 docker/install-all.sh docker/*.whl *.whl requirements.txt pyproject.toml /home/${APP_USER}/

# Install Torch with GPU support.
# RUN bash -c "/home/${APP_USER}/install-all.sh --install-miniconda --use-local-wheels"
RUN bash -c "/home/${APP_USER}/install-all.sh --install-miniconda"

ADD --chown=${APP_USER}:${APP_USER} --chmod=750 gradio_run.py /home/${APP_USER}/MagicQuill/
ADD --chown=${APP_USER}:${APP_USER} --chmod=750 docker/95-MagicQuill.sh /opt/nvidia/entrypoint.d/

RUN sed -i 's/host="127.0.0.1"/host="0.0.0.0"/' /home/${APP_USER}/MagicQuill/gradio_run.py

EXPOSE 7860

#
# No entry point required as the base image has it.
#