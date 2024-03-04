import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Date;

public class FeatureReducer extends Reducer<Text, Text, Text, Text> {
    private SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");

    private double calculateDistance(double lat1, double lon1, double lat2, double lon2) {
        final double R = 6371e3; // Earth radius in meters
        double φ1 = Math.toRadians(lat1);
        double φ2 = Math.toRadians(lat2);
        double Δφ = Math.toRadians(lat2 - lat1);
        double Δλ = Math.toRadians(lon2 - lon1);

        double a = Math.sin(Δφ / 2) * Math.sin(Δφ / 2) +
                Math.cos(φ1) * Math.cos(φ2) *
                        Math.sin(Δλ / 2) * Math.sin(Δλ / 2);
        double c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));

        return R * c; // Distance in meters
    }

    private long calculateTimeDifference(String start, String end) throws ParseException {
        Date startDate = sdf.parse(start);
        Date endDate = sdf.parse(end);
        return (endDate.getTime() - startDate.getTime()) / 1000; // Time difference in seconds
    }

    @Override
    protected void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
        // Previous values initialization for distance and time calculation
        String prevTime = "";
        double prevLat = 0.0, prevLon = 0.0;
        boolean isFirst = true;
        double totalDistance = 0.0;
        long totalTime = 0;

        for (Text value : values) {
            String[] parts = value.toString().split(",");

            // Data extraction
            double currentLat = Double.parseDouble(parts[4]);
            double currentLon = Double.parseDouble(parts[3]);
            String currentTime = parts[1];

            if (!isFirst) {
                // Compute distance and time difference
                double distance = calculateDistance(prevLat, prevLon, currentLat, currentLon);
                long timeDiff;
                try {
                    timeDiff = calculateTimeDifference(prevTime, currentTime);
                } catch (ParseException e) {
                    throw new IOException(e);
                }

                totalDistance += distance;
                totalTime += timeDiff;
            } else {
                isFirst = false;
            }

            prevLat = currentLat;
            prevLon = currentLon;
            prevTime = currentTime;
        }

        // After all values are processed, compute average speed or any other final metrics
        double averageSpeed = 0;
        if (totalTime > 0) {
            averageSpeed = totalDistance / totalTime * 3.6; // Convert m/s to km/h
        }

        // Emit final metrics for this MDN
        context.write(key, new Text("Total Distance: " + totalDistance + ", Average Speed: " + averageSpeed));
    }
}
