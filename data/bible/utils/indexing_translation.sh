LAST=$(exec find ../translations -regex "../translations/[0-9].*" -exec basename {} \; | sed 's/\([0-9]\+\).*/\1/g' | sort -n | tail -1)
[ -z "$LAST" ] && LAST=0 || LAST=$((LAST+1))

while read trans;
do
  mv ../translations/"$trans" ../translations/"$(printf "%03d" $((LAST)))"-"$(echo "$trans" | cut -d_ -f1)"
  LAST=$((1+LAST))
done < <(find ../translations -type d -regex "../translations/eng.*" -exec basename {} \;)
