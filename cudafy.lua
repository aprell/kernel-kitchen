local lpeg = require "lpeg"

local P, R, S, V = lpeg.P, lpeg.R, lpeg.S, lpeg.V
local C, Cc, Cs, Ct = lpeg.C, lpeg.Cc, lpeg.Cs, lpeg.Ct

---------------
-- Utilities --
---------------

local alpha = R ("AZ", "az")
local num = R "09"
local alphanum = alpha + num
local space = S " \t\n"
local identifier = C ((P "_" + alpha) ^ 1 * (P "_" + alphanum) ^ 0)

local function token(tok)
    return space ^ 0 * C (tok) * space ^ 0
end

local function expect(tok)
    return token(tok) / 0
end

local function begin_with(tok)
    return (C (tok) * space ^ 0) / 0
end

local function end_with(tok)
    return (space ^ 0 * C (tok)) / 0
end

local function match_until(pat)
    return C ((1 - expect(pat)) ^ 0)
end

local function anywhere(pat)
  return P { pat + 1 * V (1) }
end

local function gsub(str, pat, repl)
    return (Cs ((pat / repl + 1) ^ 0)):match(str)
end

local function apply(funcs)
    return function (source)
        for _, func in pairs(funcs) do
            source = func(source)
        end
        return source
    end
end

-------------
-- Kernels --
-------------

local cuda = {}

local kernel_params =
    expect "(" * match_until ")" * expect ")"

local shared_decls =
    match_until ")"

local kernel_body =
    match_until "END_KERNEL"

-- Pattern
local kernel = Ct (
    begin_with "KERNEL"
        * begin_with "("
            * identifier
            * expect ","
            * kernel_params
            * (end_with "," * shared_decls) ^ -1
        * end_with ")"
        * kernel_body
    * end_with "END_KERNEL"
)

-- Replacement
function cuda.kernel(kernel)
    if #kernel == 4 then
        -- Concatenate __shared__ declarations and kernel body
        kernel[3] = ("%s\n%s"):format(kernel[3], kernel[4])
        table.remove(kernel)
    end
    return ("__global__ void %s(%s) {%s\n}"):format(kernel[1], kernel[2] or "", kernel[3])
end

------------------
-- Kernel calls --
------------------

local kernel_config =
    expect "/* <<< */" * match_until "/* >>> */" * expect "/* >>> */"

local kernel_args =
    match_until ");"

-- Pattern
local kernel_call = Ct (
    identifier
    * begin_with "("
        * kernel_config
        * (expect "," * kernel_args) ^ -1
    * end_with ");"
)

-- Replacement
function cuda.kernel_call(call)
    return ("%s<<<%s>>>(%s);"):format(call[1], call[2], call[3] or "")
end

------------
-- CUDAFY --
------------

local cudafy = apply {
    function (source) return gsub(source, kernel, cuda.kernel) end,
    function (source) return gsub(source, kernel_call, cuda.kernel_call) end,
}

if #arg > 0 then
    for i = 1, #arg do
        local file = assert(io.open(arg[i]))
        print(cudafy(file:read("*all")))
        file:close()
    end
end
