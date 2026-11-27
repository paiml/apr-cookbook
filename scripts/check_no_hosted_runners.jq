# check_no_hosted_runners.jq -- the rules the gate applies to one workflow's JSON (#470).
# Run as: yq -o json '.' <file> | jq -r -s --arg file <basename> -f check_no_hosted_runners.jq
# Emits "<file> <job>" for each job whose runs-on names a hosted image, is unverifiable, or is malformed.
def hostedre: "^(ubuntu|windows|macos)-";
def exprre: "^\\s*\\$\\{\\{\\s*matrix\\.(?<k>[A-Za-z0-9_-]+)\\s*\\}\\}\\s*$";
# YAML 1.1 implicit scalars. A top-level matrix value spelled like this is a bool, null or
# number to the reader this gate replaces, so it cannot be checked: UNVERIFIABLE.
def ambiguous: test("^(y|yes|n|no|on|off|true|false|null|~|\\.nan|[-+]?\\.inf|[-+]?[0-9._]+|0[xob][0-9a-f_]+|[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}.*|[0-9]+(:[0-9]+)+)$"; "i");
# One runs-on label -> {u: unverifiable, items: the labels it can take}.
# Main coerces falsy values (null, false, [], "", 0) to {} (Python `or {}`): match it.
def orempty: if . == null or . == false or . == [] or . == "" or . == 0 then {} else . end;
def expand($mx):
  if type != "string" or (contains("${{") | not) then {u: false, items: [.]}
  elif test(exprre) then
    (capture(exprre).k) as $k
    | ($mx[$k]) as $v
    | if ($v | type) == "array"
         and ($v | all(type == "string" or type == "array"))
         and ([$v[] | select(type == "string") | select(ambiguous)] | length == 0)
      then {u: false, items: [$v[] | if type == "array" then .[] else . end]}
      else {u: true, items: []} end
  else {u: true, items: []} end;
(if length != 1 then error("one YAML document per workflow file") else .[0] end)
| (. | orempty) as $root
| (if ($root | type) == "object" then $root else error("workflow is not a mapping") end)
| (.jobs | orempty) as $jobs
| (if ($jobs | type) == "object" then $jobs else error("jobs is not a mapping") end)
| to_entries[]
| . as $e
| $e.value as $j
| (if ($j | type) != "object" then error("job \($e.key) is not a mapping") else . end)
| ($j.strategy | orempty) as $strat
| (if $strat != null and ($strat | type) != "object" then error("job \($e.key) strategy is not a mapping") else . end)
| (if ($strat.matrix | type) == "object" then $strat.matrix else {} end) as $mx
| ($j["runs-on"]) as $ro
| (if ($ro | type) == "object" then
     {ok: ((($ro | keys) - ["group", "labels"]) | length == 0),
      labels: ((($ro.labels // []) | if type == "array" then . else [.] end)
               + (if ($ro.group | type) == "string" then [$ro.group] else [] end))}
   elif ($ro | type) == "array" then {ok: true, labels: $ro}
   elif ($ro | type) == "string" then {ok: true, labels: [$ro]}
   else {ok: true, labels: []} end) as $s
| ([$s.labels[] | expand($mx)]) as $ex
| if ($s.ok | not) or any($ex[]; .u) or any($ex[].items[]; type == "string" and test(hostedre))
  then "\($file) \($e.key)" else empty end
