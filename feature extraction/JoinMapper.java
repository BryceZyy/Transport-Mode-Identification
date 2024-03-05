import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.IOException;

public class JoinMapper extends Mapper<LongWritable, Text, Text, Text> {
    @Override
    protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
        String line = value.toString();
        String[] parts = line.split(",");
        Text outputKey;
        Text outputValue;

        if (parts.length >= 10) {
            // Handling station data
            outputKey = new Text(parts[0]); // Station ID as key
            outputValue = new Text("0," + String.join(",", parts)); // Prefix with '0' to indicate station data
        } else if (parts.length == 4) {
            // Handling other data that needs to be joined
            outputKey = new Text(parts[3]); // Assuming the key for join is at index 3
            outputValue = new Text("1," + parts[0] + "," + parts[1]); // Prefix with '1' and select necessary fields
        } else {
            // Skip lines that do not match the expected formats
            return;
        }

        context.write(outputKey, outputValue);
    }
}
