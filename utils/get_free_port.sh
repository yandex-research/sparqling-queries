# from here: https://unix.stackexchange.com/a/423052
comm -23 \
<(seq "2000" "65000" | sort) \
<(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) \
| shuf | head -n 1
