#!/bin/bash

# Target URL
url="https://www.hiqa.ie/reports-and-publications/inspection-reports?field_county_value=Leitrim&term_node_tid_depth=16"

# Directory where PDFs will be saved
save_dir="../data/hiqa_pdfs"

# Create directory if it doesn't exist
mkdir -p "$save_dir"

# Fetch the content of the URL, filter out PDF links, and download each PDF
wget -qO- "$url" | grep -oP 'href="\K[^"]*\.pdf' |
while read -r pdf_link; do
    # Extract the filename from the URL
    filename="${pdf_link##*/}"

    # Decode URL-encoded characters in filename
    decoded_filename=$(printf '%b' "${filename//%/\\x}")

    # Ensure decoded filename does not start with "-e "
    decoded_filename="${decoded_filename#-e }"

    if [[ $pdf_link != http* ]]; then
        # Form the absolute URL if the link is relative
        pdf_link="https://www.hiqa.ie${pdf_link}"
    fi

    # Ensure decoded filename ends with .pdf
    if [[ ! $decoded_filename =~ \.pdf$ ]]; then
        decoded_filename="${decoded_filename}.pdf"
    fi

    # Formulate the complete save path
    save_path="$save_dir/$decoded_filename"

    echo "Downloading $pdf_link as $decoded_filename ..."
    wget -O "$save_path" "$pdf_link"
done

echo "Download completed."
