#!/bin/bash

curl http://localhost:4000/v1/chat/completions \
-H "Authorization: Bearer sk-local-master" \
-H "Content-Type: application/json" \
-d '{
	"model": "local",
	"messages": [
	{"role": "system", "content": "You are a helpful assistant."},
	{"role": "user", "content": "Hello, how are you? Tell me about yourself."}
	],
	"max_tokens": 300,
	"temperature": 0.7,
	"top_p": 0.9,
	"stream": false
}' | jq '.choices[0].message.content'

curl http://localhost:4000/v1/chat/completions \
-H "Authorization: Bearer sk-local-master" \
-H "Content-Type: application/json" \
-d '{
	"model": "local",
	"messages": [
	{"role": "system", "content": "You are a helpful assistant."},
	{"role": "user", "content": "Hello, how are you? Tell me about yourself."}
	],
	"max_tokens": 300,
	"temperature": 0.7,
	"top_p": 0.9,
	"stream": false
}'
