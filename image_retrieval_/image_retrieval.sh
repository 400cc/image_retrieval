#!/bin/bash

export device="cuda:0"

uvicorn main:app --host 0.0.0.0 --port 8000