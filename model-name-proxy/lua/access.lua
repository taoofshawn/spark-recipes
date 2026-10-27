-- access-phase hook for model-name-proxy (OpenResty).
--
-- Rewrites the `"model"` field of JSON request bodies to the backend's real
-- served name (BACKEND_MODEL env), so clients can always send "spark-llm"
-- while the backend hears a name it actually serves.
--
-- Only POST/PUT bodies are touched; bodies without a "model" key pass through
-- unchanged. GET/metrics/health are untouched (different location or method).

local cjson = require("cjson.safe")

-- This hook re-encodes the whole body, so the round-trip must be lossless.
-- By default cjson decodes [] and {} to the same empty Lua table and re-encodes
-- it as {}, which turns a tool schema's "required": [] into "required": {}.
-- Strict OpenAI backends reject that outright: sglang answers 400 "Tool 1
-- function has invalid 'parameters' schema: {} is not of type 'array'". The
-- array metatable tags decoded arrays so they re-encode as [], while objects
-- decoded from {} stay {}.
cjson.decode_array_with_array_mt(true)

ngx.req.read_body()
local raw = ngx.req.get_body_data()

-- Large bodies spill to disk; read the file in that case.
if not raw then
    local path = ngx.req.get_body_file()
    if path then
        local f = io.open(path, "rb")
        if f then raw = f:read("*a"); f:close() end
    end
end

if not raw or raw == "" then return end

local decoded = cjson.decode(raw)
if not decoded or type(decoded) ~= "table" or decoded.model == nil then
    return  -- pass through untouched
end

local backend = os.getenv("BACKEND_MODEL")
if not backend or backend == "" then return end

-- Already normalized (client sent the real served name): leave the bytes
-- alone rather than round-tripping them through cjson for no reason.
if decoded.model == backend then return end

decoded.model = backend
local ok, encoded = pcall(cjson.encode, decoded)
if not ok or not encoded then return end  -- re-encode failure: pass through

ngx.req.set_body_data(encoded)
