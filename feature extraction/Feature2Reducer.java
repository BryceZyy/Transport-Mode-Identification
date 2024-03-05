import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class Feature2Reducer extends Reducer<Text, Text, Text, Text> {
    
    @Override
    protected void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
        List<String> lines = new ArrayList<>();
        values.forEach(value -> lines.add(value.toString()));

        if (lines.size() > 3) {
            for (String line : lines) {
                context.write(key, new Text(line));
            }
        }
    }
}
