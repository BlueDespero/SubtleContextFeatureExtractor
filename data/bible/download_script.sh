while read line;
do
	wget https://ebible.org/Scriptures/"$line" && \
	unzip "$line" -d $(basename "$line" .zip) && \
	rm -f "$line"
done < <(awk 'match($0, /<a href.*<\/a>/) { print substr( $0, RSTART, RLENGTH )}' bibles_page_html.txt | cut -d\" -f2 | grep -E "^eng.*readaloud.zip")
