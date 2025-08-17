import streamlit as st
import json
import pandas as pd
from shapely.geometry import Polygon

st.title("Area Calculator Using Roboflow Annotations")

uploaded_file = st.file_uploader("Choose a COCO JSON file", type="json")

if uploaded_file is not None:
    try:
        # Load the COCO JSON data (if using uploader)
        coco_data = json.load(uploaded_file)

        # Create lookup dictionaries
        image_id_to_filename = {img['id']: img['file_name']
                                for img in coco_data['images']}
        category_id_to_name = {cat['id']: cat['name']
                               for cat in coco_data['categories']}

        # List to store data for the DataFrame
        raw_results_list = []

        # Iterate through annotations
        for annotation in coco_data['annotations']:
            image_id = annotation['image_id']
            category_id = annotation['category_id']

            # Ensure it's a polygon segmentation and not RLE
            if 'segmentation' in annotation and annotation['iscrowd'] == 0:
                total_object_area = 0

                # Handle cases where an object has multiple polygons
                for segment in annotation['segmentation']:
                    # Reshape the list of coordinates into a list of (x, y) tuples
                    polygon_coords = [(segment[i], segment[i+1])
                                      for i in range(0, len(segment), 2)]

                    # Create a Shapely Polygon
                    polygon = Polygon(polygon_coords)

                    # Calculate and add to the total area for the object
                    total_object_area += polygon.area

                filename = image_id_to_filename.get(image_id, 'Unknown File')
                object_name = category_id_to_name.get(
                    category_id, 'Unknown Object')

                # Append the data for each object to the list
                raw_results_list.append({
                    "File Name": filename,
                    "Object Name": object_name,
                    "Total Area": total_object_area
                })

        # Convert the raw results to a Pandas DataFrame
        df_raw_results = pd.DataFrame(raw_results_list)

        st.subheader("Calculated Area: ")

        if not df_raw_results.empty:
            # Group by "File Name" and "Object Name" and sum the "Total Area"
            df_aggregated_results = df_raw_results.groupby(["File Name", "Object Name"])[
                "Total Area"].sum().reset_index()

            # Display as an interactive DataFrame
            st.dataframe(df_aggregated_results, use_container_width=True)
        else:
            st.warning("No polygon annotations found in the uploaded file.")

    except json.JSONDecodeError:
        st.error(
            "Error: Could not decode JSON from the uploaded file. Please ensure it's a valid COCO JSON format.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
