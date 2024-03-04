import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class FeatureComputationJob {
    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            System.err.println("Usage: FeatureComputationJob <input path> <output path>");
            System.exit(-1);
        }

        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "Mobile Data Feature Computation");

        job.setJarByClass(FeatureComputationJob.class);
        job.setMapperClass(FeatureMapper.class);
        job.setReducerClass(FeatureReducer.class);

        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        boolean jobCompletedSuccessfully = job.waitForCompletion(true);
        System.exit(jobCompletedSuccessfully ? 0 : 1);
    }
}
