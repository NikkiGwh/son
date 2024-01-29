"""
This file contains the standard figure sizes (in mm) for the IEEE 2 column layout.
Width are defined as multiples of the column width.
Height are defined as fractions of the column width or the page height.

"""


# Standard values from IEEE 2 column layout in mm
column_width = 88.9
page_height = 234.9
column_separator = 6.35
thesis_text_width = 159

dimensions_by_type = {}

figure_widths = {
    "SC": 1 * column_width, 
    "HC": 1.5 * column_width,
    "DC": 2 * column_width + column_separator,
    "TH": thesis_text_width
}

figure_heights = {
    "2-3rd-CW": 2/3 * column_width,
    "3-4th-CW": 3/4 * column_width,
    "3-3rd-CW": 1 * column_width,
    "1-3rd-PH": 1/3 * page_height,
    "3-2nd-CW": 3/2 * column_width,
    "2-1st-CW": 2 * column_width,
    
    "1-1-TH": thesis_text_width,
    "3-4-TH": thesis_text_width* 3/4,
    "2-3-TH": thesis_text_width * 2/3,
    "1-3-TH": thesis_text_width * 1/3,
}

for width_type, width  in figure_widths.items():
    for height_type, height in figure_heights.items(): 
        dimensions_by_type[f"{width_type}--{height_type}"] = {
            "height": round(height, ndigits=1),
            "width": round(width, ndigits=1),
            }

