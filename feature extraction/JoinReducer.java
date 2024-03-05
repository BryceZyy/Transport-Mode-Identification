import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class JoinReducer extends Reducer<Text, Text, Text, Text> {
    @Override
    protected void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
        String stationInfo = null;
        List<String> otherInfo = new ArrayList<>();

        for (Text value : values) {
            String currentValue = value.toString();
            if (currentValue.startsWith("0,")) {
                // This is station data
                stationInfo = currentValue.substring(2); // Remove prefix
            } else if (currentValue.startsWith("1,")) {
                // This is the other data to be joined
                otherInfo.add(currentValue.substring(2)); // Remove prefix
            }
        }

        if (stationInfo != null && !otherInfo.isEmpty()) {
            for (String info : otherInfo) {
                context.write(new Text(info), new Text(stationInfo));
            }
        }
    }
}
