-- access-phase hook for model-name-proxy (OpenResty).
--
-- Rewrites the `"model"` field of JSON request bodies to the backend's real
-- served name (BACKEND_MODEL env), so clients can always send "spark-model"
-- while vLLM hears a name it actually serves.
--
-- Only POST/PUT bodies are touched; bodies without a "model" key pass through
-- unchanged. GET/metrics/health are untouched (different location or method).

local cjson = require("cjson.safe")

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

decoded.model = backend
local ok, encoded = pcall(cjson.encode, decoded)
if not ok or not encoded then return end  -- re-encode failure: pass through

ngx.req.set_body_data(encoded)
