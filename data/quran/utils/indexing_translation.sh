LAST=$(exec find ../translations -regex "../translations/[0-9].*" -exec basename {} \; | sed 's/\([0-9]\+\).*/\1/g' | sort -n | tail -1)
LAST=${LAST:-0}

while read trans;
do
  LAST=$((1+LAST))
  mv ../translations/"$trans" ../translations/"$(printf "%03d" $(( 1 + LAST)))"-"$(echo "$trans" | cut -d- -f2)"
done < <(ls ../translations/eng*.txt)
