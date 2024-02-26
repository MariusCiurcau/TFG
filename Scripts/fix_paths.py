import os
import unicodedata

def remove_accents_and_spaces(folder_path):
    # Iterate over all files in the given folder
    for filename in os.listdir(folder_path):
        # Check if the file is a normal file and not a directory
        if os.path.isfile(os.path.join(folder_path, filename)):
            # Normalize the unicode string to decompose characters with accents into base characters
            normalized_filename = unicodedata.normalize('NFD', filename)
            # Encode to ASCII bytes, then decode back to string ignoring non-ASCII characters
            ascii_filename = normalized_filename.encode('ascii', 'ignore').decode('ascii')
            # Replace spaces with underscores
            new_filename = ascii_filename.replace(' ', '_')
            # Rename the file if the new filename is different from the original
            if new_filename != filename:
                original_path = os.path.join(folder_path, filename)
                new_path = os.path.join(folder_path, new_filename)
                os.rename(original_path, new_path)
                print(f'Renamed "{filename}" to "{new_filename}"')

# Replace '/path/to/folder' with the path to the folder containing the files to be renamed

remove_accents_and_spaces("../Datasets/FXMalaga/images")
