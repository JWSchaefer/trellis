# Run all example scripts
run-examples:
    #!/usr/bin/env nu
    let results = (ls examples/*.py | get name | each { |f|
        print $"(ansi blue)Running ($f)...(ansi reset)"
        try {
            uv run python $f
            print $"(ansi green)✓ ($f)(ansi reset)\n"
            {file: $f, passed: true}
        } catch {
            print $"(ansi red)✗ ($f)(ansi reset)\n"
            {file: $f, passed: false}
        }
    })
    let failed = ($results | where passed == false | length)
    if $failed > 0 {
        error make {msg: $"($failed) examples failed"}
    }
