import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.IOException;

public class FeatureMapper extends Mapper<LongWritable, Text, Text, Text> {

    @Override
    protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
        // Assuming the input format is correct and does not need preprocessing
        String[] parts = value.toString().split(",");
        // Use MDN as key, and the entire line as value
        context.write(new Text(parts[0]), value);
    }
}
