#!/bin/bash
. ./venv/bin/activate
export AIRFLOW__API__AUTH_BACKEND='airflow.api.auth.backend.basic_auth'
airflow standalone