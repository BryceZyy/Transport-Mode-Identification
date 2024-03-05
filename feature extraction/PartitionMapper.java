import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.IOException;

public class PartitionMapper extends Mapper<LongWritable, Text, Text, Text> {
    @Override
    protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
        String[] parts = value.toString().split("\\|");
        if (parts.length > 8) {
            Text outputKey = new Text(parts[1]); // Assuming this is the key
            Text outputValue = new Text(parts[0] + "\t" + parts[7] + "\t" + parts[8]); // Concatenating as value
            context.write(outputKey, outputValue);
        }
    }
}
