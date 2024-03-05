import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import java.io.IOException;

public class Feature2Mapper extends Mapper<LongWritable, Text, Text, Text> {

    private Text outKey = new Text();
    private Text outValue = new Text();
    private String premdn = "";
    private int flag1 = 0;
    private int flag2 = 0;
    private int flag3 = 0;
    private String[] mdns = new String[3];
    private String[] times = new String[3];
    private String[] cids = new String[3];
    private String[] longs = new String[3];
    private String[] lats = new String[3];
    private String[] vals = new String[3];
    private String[] durations = new String[3];
    private String[] distances = new String[3];
    private String[] speeds = new String[3];

    @Override
    protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
        String[] parts = value.toString().split(",");
        String mdn = parts[0];
        String time = parts[1];
        String cid = parts[2];
        String longi = parts[3];
        String lat = parts[4];
        String val = String.join(",", parts[5], parts[6], parts[7], parts[8], parts[9], parts[10], parts[11], parts[12], parts[13], parts[14], parts[15], parts[16], parts[17]);
        String duration = parts[18];
        String distance = parts[19];
        String speed = parts[20];

        if (!mdn.equals(premdn)) {
            if (!premdn.isEmpty()) {
                if (flag1 == 1 && flag2 == 0) {
                    context.write(new Text(mdns[0]), new Text(String.join(",", times[0], cids[0], longs[0], lats[0], vals[0], durations[0], distances[0], speeds[0], "0.0", "-1.0")));
                } else if (flag2 == 1 && flag3 == 0) {
                    double acc = accel(Double.parseDouble(speeds[0]), Double.parseDouble(speeds[1]), Double.parseDouble(durations[1]));
                    context.write(new Text(mdns[1]), new Text(String.join(",", times[1], cids[1], longs[1], lats[1], vals[1], durations[1], distances[1], speeds[1], String.valueOf(acc), "-1.0")));
                } else if (flag3 == 1) {
                    double cos2 = cosangle(Double.parseDouble(lats[0]), Double.parseDouble(longs[0]), Double.parseDouble(lats[1]), Double.parseDouble(longs[1]), Double.parseDouble(lats[2]), Double.parseDouble(longs[2]));
                    double acc2 = accel(Double.parseDouble(speeds[0]), Double.parseDouble(speeds[1]), Double.parseDouble(durations[1]));
                    context.write(new Text(mdns[1]), new Text(String.join(",", times[1], cids[1], longs[1], lats[1], vals[1], durations[1], distances[1], speeds[1], String.valueOf(acc2), String.valueOf(cos2))));
                    double cos3 = -1;
                    double acc3 = accel(Double.parseDouble(speeds[1]), Double.parseDouble(speeds[2]), Double.parseDouble(durations[2]));
                    context.write(new Text(mdns[2]), new Text(String.join(",", times[2], cids[2], longs[2], lats[2], vals[2], durations[2], distances[2], speeds[2], String.valueOf(acc3), String.valueOf(cos3))));
                }
                context.write(new Text(mdns[0]), new Text(String.join(",", times[0], cids[0], longs[0], lats[0], vals[0], durations[0], distances[0], speeds[0], "0.0", "1.0")));
            }
            for (int i = 0; i < 3; i++) {
                mdns[i] = mdn;
                times[i] = time;
                cids[i] = cid;
                longs[i] = longi;
                lats[i] = lat;
                vals[i] = val;
                durations[i] = duration;
                distances[i] = distance;
                speeds[i] = speed;
            }
            premdn = mdn;
            flag1 = 1;
        } else {
            mdns[0] = mdns[1];
            mdns[1] = mdns[2];
            mdns[2] = mdn;
            times[0] = times[1];
            times[1] = times[2];
            times[2] = time;
            cids[0] = cids[1];
            cids[1] = cids[2];
            cids[2] = cid;
            longs[0] = longs[1];
            longs[1] = longs[2];
            longs[2] = longi;
            lats[0] = lats[1];
            lats[1] = lats[2];
            lats[2] = lat;
            vals[0] = vals[1];
            vals[1] = vals[2];
            vals[2] = val;
            durations[0] = durations[1];
            durations[1] = durations[2];
            durations[2] = duration;
            distances[0] = distances[1];
            distances[1] = distances[2];
            distances[2] = distance;
            speeds[0] = speeds[1];
            speeds[1] = speeds[2];
            speeds[2] = speed;
            if (flag1 == 0) {
                flag1 = 1;
            } else if (flag2 == 0) {
                flag2 = 1;
            } else {
                flag3 = 1;
            }
        }
    }
}
