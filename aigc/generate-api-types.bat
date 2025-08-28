@echo off
echo Generating API types from OpenAPI spec...

REM 使用 npx 运行 openapi-typescript 并输出到 api-types.ts
npx openapi-typescript http://47.110.156.41:7000/openapi.json -o api-types.ts

echo Done.
pause