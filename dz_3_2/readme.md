## Запуск LiteLLM:

**Необходимые для работы сервера параметры (в проде установим в .env)**

$env:GIGACHAT_CREDENTIALS = "<api_key>"

$env:GIGACHAT_SCOPE = "GIGACHAT_API_PERS"

litellm --config ./dz_3_2/config.yaml --port 4000 --debug

## Отправка запроса:

curl.exe -X POST http://localhost:4000/v1/chat/completions -H "Content-Type: application/json" -d @dz_3_2/request.json