# average_profile.py

# This script reads the 'profile.langevin' file, processes the data to compute
# the average number of atoms and average temperature for each bin across all timesteps,
# and writes the results to 'data.txt'.

def main():
    # Initialize a dictionary to store data for each bin
    bin_data = {}

    input_filename = 'profile.langevin'
    output_filename = 'data.txt'

    with open(input_filename, 'r') as file:
        # Skip the first three comment lines
        for _ in range(3):
            next(file)

        while True:
            try:
                # Read the timestep header line
                line = next(file).strip()
            except StopIteration:
                # End of file reached
                break

            # Skip empty lines or comments
            if not line or line.startswith('#'):
                continue

            # Parse the timestep line
            parts = line.split()
            if len(parts) != 3:
                continue  # Invalid timestep line, skip

            timestep = int(parts[0])
            num_chunks = int(float(parts[1]))  # Handle float format
            total_count = float(parts[2])

            # Read data for each chunk/bin in this timestep
            for _ in range(num_chunks):
                try:
                    data_line = next(file).strip()
                except StopIteration:
                    print("Unexpected end of file.")
                    return

                # Skip empty lines or comments
                if not data_line or data_line.startswith('#'):
                    continue

                data_parts = data_line.split()
                if len(data_parts) != 4:
                    continue  # Invalid data line, skip

                # Extract data for the bin
                try:
                    bin_serial = int(data_parts[0])
                    bin_position = float(data_parts[1])
                    ncount = float(data_parts[2])
                    v_temp = float(data_parts[3])
                except ValueError:
                    continue  # Invalid data values, skip

                # Initialize bin entry if it doesn't exist
                if bin_serial not in bin_data:
                    bin_data[bin_serial] = {
                        'bin_position': bin_position,
                        'ncount_list': [],
                        'v_temp_list': []
                    }

                # Append the data to the bin's lists
                bin_data[bin_serial]['ncount_list'].append(ncount)
                bin_data[bin_serial]['v_temp_list'].append(v_temp)

    # Compute averages for each bin
    with open(output_filename, 'w') as outfile:
        # Start writing data from the first line without any header
        # Process bins in order of their serial numbers
        for bin_serial in sorted(bin_data.keys()):
            bin_info = bin_data[bin_serial]
            bin_position = bin_info['bin_position']
            ncount_avg = sum(bin_info['ncount_list']) / len(bin_info['ncount_list'])
            v_temp_avg = sum(bin_info['v_temp_list']) / len(bin_info['v_temp_list'])

            # Write the averaged data to the output file
            outfile.write(f'{bin_serial} {bin_position} {ncount_avg} {v_temp_avg}\n')

if __name__ == '__main__':
    main()
